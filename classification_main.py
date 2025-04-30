import torch
from densenet import densenet169_classifier
from train import train_classification_model, get_classification_metrics, k_means_clustering, plot_pca
from pipeline import get_classification_data, get_dataloaders

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def main(train=True, subsample_rate=None, checkpoint_path=None):
    study_data = get_classification_data(subsample_rate=subsample_rate)
    num_classes = 7
    
    #### Create dataloaders pipeline
    data_cat = ['train', 'valid']  # data categories
    dataloaders = get_dataloaders(study_data, batch_size=16)
    dataset_sizes = {x: len(study_data[x]) for x in data_cat}

    #### Build model
    model = densenet169_classifier(pretrained=True, num_classes=num_classes)

    for param in list(model.parameters())[:-2]:
        param.requires_grad = False
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.1)

    #### Train model
    if train:
        model = train_classification_model(model, criterion, optimizer, dataloaders, scheduler,
                            dataset_sizes, num_epochs=20, num_classes=num_classes)
        torch.save(model.state_dict(), 'models/model_classification.pth')
    else:
        model = densenet169_classifier(pretrained=True, num_classes=num_classes)
        model.load_state_dict(torch.load('models/model_classification.pth'))
        model.training = False
        model.to(device)

    vector_data = get_classification_metrics(model, criterion, dataloaders, dataset_sizes, num_classes, phase='valid')
    cluster_centers = k_means_clustering(vector_data, num_classes)
    plot_pca(vector_data, cluster_centers)

if __name__ == '__main__':
    main(train=False, subsample_rate=0.1)
