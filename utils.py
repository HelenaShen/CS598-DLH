import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchnet import meter

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def plot_training(costs, accs):
    '''
    Plots curve of Cost vs epochs and Accuracy vs epochs for 'train' and 'valid' sets during training
    '''
    # Convert tensors to CPU and then to numpy arrays
    train_acc = [acc.cpu().numpy() for acc in accs['train']]
    valid_acc = [acc.cpu().numpy() for acc in accs['valid']]
    train_cost = [cost.cpu().detach().numpy() for cost in costs['train']]
    valid_cost = [cost.cpu().detach().numpy() for cost in costs['valid']]
    epochs = range(len(train_acc))

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_cost)
    plt.plot(epochs, valid_cost)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Cost')
    
    plt.show()

def n_p(x):
    '''convert numpy float to Variable tensor float'''    
    return Variable(torch.tensor([x], dtype=torch.float32, device=device), requires_grad=False)
    # return Variable(torch.cuda.FloatTensor([x]), requires_grad=False)

def get_count(df, cat):
    '''
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for abnormal and "negative" for normal
    '''
    return df[df['Path'].str.contains(cat)]['Count'].sum()


if __name__=='main':
    pass