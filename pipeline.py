import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

data_cat = ['train', 'valid'] # data categories
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_study_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes, 
    these dataframes contain three columns 'Path', 'Count', 'Label'
    Args:
        study_type (string): one of the seven study type folder names in 'train/valid/test' dataset 
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        BASE_DIR = 'MURA-v1.1/%s/%s/' % (phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1] # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in tqdm(patients): # for each patient folder
            for study in os.listdir(BASE_DIR + patient): # for each study in that patient folder
                label = study_label[study.split('_')[1]] # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/' # path to this study
                valid_images = [f for f in os.listdir(path) if not f.startswith('.') and f.endswith('.png')]
                study_data[phase].loc[i] = [path, len(valid_images), label] # add new row
                i+=1
    return study_data

def get_patient_level_csv_data(study_type):
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    study_csv_paths = {
        'train': 'MURA-v1.1/train_image_paths.csv',
        'valid': 'MURA-v1.1/valid_image_paths.csv',
    }
    for phase in study_csv_paths.keys():
        df = pd.read_csv(study_csv_paths[phase], header=None, names=['Path'])
        df['Study'] = df['Path'].apply(lambda x: x.split('/')[2])
        df['Label'] = df['Path'].apply(lambda x: study_label[x.split('/')[-2].split('_')[1]])
        df['Count'] = 1
        study_data[phase] = df[df['Study'] == study_type]
    return study_data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['Path'].iloc[idx]
        images = self.transform(pil_loader(image_path))
        label = self.df['Label'].iloc[idx]
        sample = {'images': images, 'label': label}
        return sample

def get_dataloaders(data, batch_size=32, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size//2, shuffle=False, num_workers=0)
    }
    return dataloaders

if __name__ == '__main__':
    # study_data = get_study_level_data(study_type='XR_WRIST')
    # dataloaders = get_dataloaders(study_data, batch_size=1)
    study_data = get_patient_level_csv_data(study_type='XR_WRIST')
    dataloaders = get_dataloaders(study_data, batch_size=1, study_level=True)