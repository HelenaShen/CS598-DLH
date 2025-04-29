import time
import copy
import torch
from torchnet import meter
from utils import plot_training
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
data_cat = ['train', 'valid']  # data categories


def train_model(model, criterion, optimizer, dataloaders, scheduler,
                dataset_sizes, num_epochs, study_type, start_epoch=0):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x: [] for x in data_cat}  # for storing costs per epoch
    accs = {x: [] for x in data_cat}  # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(start_epoch, num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=True)
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase == 'train')
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                print(i, end='\r')
                if phase == 'valid':
                    with torch.no_grad():
                        inputs = data['images'].to(device)
                        labels = data['label'].type(torch.float32).to(device)
                        outputs = model(inputs)
                else:
                    inputs = data['images'].to(device)
                    labels = data['label'].type(torch.float32).to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                loss = criterion(outputs, labels, phase)
                loss = loss.mean()
                running_loss += loss
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = (outputs.data > 0.5).float().flatten()
                running_corrects += torch.sum(preds == labels.data)
                confusion_matrix[phase].add(preds.cpu(), labels.data.cpu())
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print('Confusion Meter:\n', confusion_matrix[phase].value())
            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            # At the end of each epoch
            print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        if (epoch + 1) % 10 == 0:
            # save checkpoint
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'costs': costs,
                'accs': accs
            }, f'checkpoints/{study_type}_checkpoint_{epoch+1}.pth')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    '''
    Loops over phase (train or valid) set to determine acc, loss and 
    confusion meter of the model.
    '''
    confusion_matrix = meter.ConfusionMeter(2, normalized=True)
    running_loss = 0.0
    running_corrects = 0
    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        with torch.no_grad():
            labels = data['label'].type(torch.float32).to(device)
            inputs = data['images'].to(device)
            # forward
            outputs = model(inputs)
            # outputs = torch.mean(outputs)
            loss = criterion(outputs, labels, phase)
            loss = loss.mean()
            # statistics
            running_loss += loss
            preds = (outputs.data > 0.5).float().flatten()
            running_corrects += torch.sum(preds == labels.data)
            confusion_matrix.add(preds.cpu(), labels.data.cpu())

    loss = running_loss / dataset_sizes[phase]
    acc = running_corrects / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    print('Confusion Meter:\n', confusion_matrix.value())


def get_pr_curve(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    """use sklearn to plot precision recall curve"""
    # Get predictions and labels
    predictions = []
    labels = []
    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        with torch.no_grad():
            label = data['label'].type(torch.float32).to(device)
            input = data['images'].to(device)
            output = model(input)
            pred = output.data.float().flatten()
            predictions.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(labels, predictions)

    # Plot precision-recall curve
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
