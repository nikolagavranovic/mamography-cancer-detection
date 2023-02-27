import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torchvision import transforms
from custom_dataset import RSNAMamographyDataset
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, WeightedRandomSampler

def make_datasets(train_csv, imgs_dir, val_pct):

    # make augmentator and dataset
    augmentator = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(5),
        transforms.ToTensor(), # return it as a tensor and transforms it to [0, 1]
    ])
    dataset = RSNAMamographyDataset(train_csv, imgs_dir, augmentator)

    # split train and test dataset
    val_size = int(val_pct * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def make_dataloaders(train_csv, train_dataset, val_dataset, batch_size):
    dftrain = pd.read_csv(train_csv)
    
    print("Class counting...")
    labels = dftrain['cancer'].values
    class_sample_count = np.array([len(np.where(labels == l)[0]) for l in np.unique(labels)])


    # since there is big class imbalance, we will not sample positive class THAT frequent
    # to be closer to 'reality, every fifth image will be cancer (instead of 50/50 distribution)'
    class_sample_count[1] *= 5
    class_weights = 1. / class_sample_count

    print("Adding weights to each training sample...")
    sample_weights = [class_weights[label] for _, label in tqdm(train_dataset)]

    sample_weights = np.array(sample_weights)
    sample_weights = torch.from_numpy(sample_weights)
    weighted_random_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    
    # Applying random sampler just tu train dataset, not for validation, since the validation dataset should be imitation of 'real' DS
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = 2, pin_memory = True, sampler = weighted_random_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True, pin_memory = True)
    
    dataloaders = {'train' : train_dataloader, 'val' : val_dataloader}
