
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


from training_helpers import find_optim_thres

class CancerClasificatior(nn.Module):
    def __init__(self, n = 5):
        super(CancerClasificatior, self).__init__()
        self.network = models.resnext50_32x4d(pretrained = True)
        
        
        # freeze first n layers
        ct = 0
        if n:
            print('FREEZED LAYERS:')
        for child in self.network.children():
            ct += 1
            if ct <= n:
                print('*'*100)
                print(child)

                for param in child.parameters():
                    param.requires_grad = False

        n_features = self.network.fc.out_features
        # add additional layer that maps 2048 extracted features from resnet to 1 feature determining the class
        self.classifier_layer = nn.Sequential(
            nn.Linear(n_features , 64),
            nn.Dropout(0.3),
            nn.Linear(64 , 1)
        )
    
    def forward(self, xb):        
        xb = self.network(xb)
        xb = self.classifier_layer(xb)
        return torch.sigmoid(xb)
    

def train_model(model, criterion, optimizer, scheduler, earlystopper, checkpoint, device, dataloaders, num_epochs=25):
    since = time.time()
    
    metricf1 = BinaryF1Score()
    precision = BinaryPrecision()
    recall = BinaryRecall()
    accuracy = BinaryAccuracy()
    roc = BinaryROC()
    auc = BinaryAUROC()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    
    train_metrics = {'loss' : [], 'acc' : [], 'f1': [], 'precision': [], 'recall': [], 'auc': [], 'lr': []}
    val_metrics = {'loss' : [], 'acc' : [], 'f1': [], 'precision': [], 'recall': [], 'auc': []}
    
    
    # inital threshold for first epoch, it will change afterwards
    threshold = 0.5
    
    print('Starting training...')
    print('-' * 20)
    for epoch in range(num_epochs):
        

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # empty 'all' tensors for saving
            # for calculating aoc at the end of epoch, and for calculating new threshold
            all_outputs = torch.Tensor([])
            all_labels = torch.Tensor([])
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            n_samples = 0
            
            n_correct = 0
            running_f1 = 0.0
            # Iterate over data.
            print(f'{phase} for epoch {epoch + 1}')
            for inputs, labels in tqdm(dataloaders[phase]):
                
                labels = torch.unsqueeze(labels.to(torch.float32), 1)
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (outputs > threshold).double()
                    #print(all_outputs)
                    #print(outputs)
                    # concatenating all outputs and labels for calculation aoc and new threshold
                    all_outputs = torch.cat((all_outputs, outputs.to('cpu')))
                    all_labels = torch.cat((all_labels, labels.to('cpu')))
                    
                    #print(labels)
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                # n_samples += labels.size(0)
                running_loss += loss.item()
                # n_correct += (preds == labels).sum().item()
                # running_f1 += metric(outputs, labels) 


                # collect any unused memmory
                gc.collect()
                torch.cuda.empty_cache()
            
            # statistics
            epoch_loss = running_loss / len(dataloaders[phase])
            
            # find true positive and false positive rates for ROC curve
            fpr, tpr, thresholds = roc(all_outputs, all_labels)
            epoch_auc = auc(all_outputs, all_labels)
            # find new threshold
            if phase == 'train':
                # just for train phase calculate threshold
                threshold, _ = find_optim_thres(fpr, tpr, thresholds)
                print(f'New threshold is {threshold}')
            # calculate metrics using new optimized threshold
            epoch_f1 = metricf1(all_outputs > threshold, all_labels)
            epoch_acc = accuracy(all_outputs > threshold, all_labels)
            epoch_precision = precision(all_outputs > threshold, all_labels)
            epoch_recall = recall(all_outputs > threshold, all_labels)
            
            # save all of the statistics for latter analysis
            if phase == 'train':
                train_metrics['lr'].append(optimizer.param_groups[0]['lr'])
                scheduler.step()
                train_metrics['loss'].append(epoch_loss)
                train_metrics['acc'].append(epoch_acc)
                train_metrics['f1'].append(epoch_f1)
                train_metrics['precision'].append(epoch_precision)
                train_metrics['recall'].append(epoch_recall)
                train_metrics['auc'].append(epoch_auc)


            else:
                val_metrics['loss'].append(epoch_loss)
                val_metrics['acc'].append(epoch_acc)
                val_metrics['f1'].append(epoch_f1)
                val_metrics['precision'].append(epoch_precision)
                val_metrics['recall'].append(epoch_recall)
                val_metrics['auc'].append(epoch_auc)



            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                # calclulate new threshold and save model
                checkpoint['threshold'] = threshold
                torch.save(checkpoint, 'checkpoint.pth')

                
        # cant be formated in string
        tr_loss, tr_acc, tr_f1, tr_prec, tr_rec, tr_auc = train_metrics['loss'][-1], train_metrics['acc'][-1],  train_metrics['f1'][-1], train_metrics['precision'][-1], train_metrics['recall'][-1], train_metrics['auc'][-1]
        val_loss, val_acc, val_f1, val_prec, val_rec, val_auc = val_metrics['loss'][-1], val_metrics['acc'][-1], val_metrics['f1'][-1], val_metrics['precision'][-1], val_metrics['recall'][-1], val_metrics['auc'][-1]
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{num_epochs}, learning rate: {lr}')
        print(f'Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}, Train f1: {tr_f1:.4f}, Train Precision: {tr_prec:.4f}, Train Recall: {tr_rec:.4f}, Train AUC: {tr_auc:.4f}')
        print(f'Valitadion Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, Vall f1: {val_f1:.4f}, Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val AUC: {val_auc:.4f}')
        
        if earlystopper.early_stop(val_loss):
            break
        
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val f1: {best_f1:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_metrics, val_metrics