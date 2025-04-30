import torch
from densenet import densenet169
from utils import n_p, get_count
from train import train_model, get_metrics
from pipeline import get_dataloaders, get_study_level_csv_data

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main(train=True, study_type='XR_WRIST', checkpoint_path=None, subsample_rate=None):
    # #### load study level dict data
    study_data = get_study_level_csv_data(study_type=study_type, subsample_rate=subsample_rate)
    study_type = study_type if study_type is not None else 'all'

    # #### Create dataloaders pipeline
    data_cat = ['train', 'valid']  # data categories
    dataloaders = get_dataloaders(study_data, batch_size=16)
    dataset_sizes = {x: len(study_data[x]) for x in data_cat}

    # #### Build model
    tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
    Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
    Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

    print('tai:', tai)
    print('tni:', tni, '\n')
    print('Wt0 train:', Wt0['train'])
    print('Wt0 valid:', Wt0['valid'])
    print('Wt1 train:', Wt1['train'])
    print('Wt1 valid:', Wt1['valid'])

    class Loss(torch.nn.modules.Module):
        def __init__(self, Wt1, Wt0):
            super(Loss, self).__init__()
            self.Wt1 = Wt1
            self.Wt0 = Wt0

        def forward(self, inputs, targets, phase):
            epsilon = 1e-7
            inputs = torch.clamp(inputs, epsilon, 1-epsilon)
            loss = - (self.Wt1[phase] * targets * torch.log(inputs.flatten()) +
                      self.Wt0[phase] * (1 - targets) * torch.log(1 - inputs.flatten()))
            return loss

    # Freeze early layers if using pretrained
    model = densenet169(pretrained=True)
    # Freeze first few layers
    # Keep last few layers trainable
    for param in list(model.parameters())[:-4]:
        param.requires_grad = False
    model.to(device)

    criterion = Loss(Wt1, Wt0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.1)

    # Train model
    if train:
        start_epoch = 0
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            print(f"Resuming from epoch {start_epoch} with best accuracy: {best_acc}")
        model = train_model(model, criterion, optimizer,
                            dataloaders, scheduler, dataset_sizes, num_epochs=20,
                            study_type=study_type, start_epoch=start_epoch)
        torch.save(model.state_dict(), f'models/model_{study_type}.pth')
    else:
        model = densenet169(pretrained=True)
        model.load_state_dict(torch.load(f'models/model_{study_type}.pth'))
        model.to(device)

    get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid')
    # get_pr_curve(model, criterion, dataloaders, dataset_sizes, phase='valid')

if __name__ == '__main__':
    main(train=True, study_type=None, subsample_rate=0.2)
