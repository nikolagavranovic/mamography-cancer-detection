import os
import time
import numpy as np
import pandas as pd
# image manipulation
import cv2
import PIL
from PIL import Image

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# helpers
from tqdm.notebook import tqdm
import time
import copy
import gc
from enum import Enum


# for cnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR
from torchvision import models
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy, BinaryROC, BinaryAUROC, BinaryPrecisionRecallCurve
from torchvision import transforms
from warmup_scheduler_pytorch import WarmUpScheduler

from custom_dataset import RSNAMamographyDataset
from model import CNN, train_model
from training_helpers import BCELoss_class_weighted, EarlyStopper, find_optim_thres
from data_helpers import make_datasets, make_dataloaders

if __name__ == '__main__':

    # Configure paths to csv file and images. Images can be downloaded from link in readme.
    train_csv = 'csvpath'
    imgs_dir = 'imgspath' 

    dftrain = pd.read_csv(train_csv)

    val_pct = 0.1
    train_dataset, val_dataset = make_datasets(train_csv, imgs_dir, val_pct)


    batch_size = 32
    dataloaders = make_dataloaders(train_csv, train_dataset, val_dataset, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Current device is {device}')


    # defining the model
    model = CNN()
    model.to(device)


    # Try this initial LR
    init_lr = 4e-05
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=init_lr)

    w_pos = 5
    w_neg = 1
    print(f"Class weight for negative class: {w_neg}, and for positive {w_pos}")
    criterion = BCELoss_class_weighted(weights = [w_neg, w_pos])
    #criterion = nn.BCEWithLogitsLoss()
    # define early stopping
    earlystopper = EarlyStopper(patience = 3)


    checkpoint = {'model': CNN(),
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'threshold' : 0.5,
            'batch_size': batch_size}
    
    n_epochs = 8
    #lr_scheduler = ExponentialLR(optimizer, gamma = 0.85, last_epoch=- 1, verbose=False)
    lr_scheduler = CosineAnnealingLR(optimizer, n_epochs, last_epoch = -1)
    # define lr scheduler
    warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                    warmup_steps=1,
                                    warmup_start_lr=init_lr/10,
                                    warmup_mode='linear')
    
    model, train_metrics, val_metrics = train_model(model, criterion, optimizer, warmup_scheduler, earlystopper, checkpoint, dataloaders,num_epochs=n_epochs)


