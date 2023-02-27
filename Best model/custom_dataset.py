import os
import time
import numpy as np
import pandas as pd
# image manipulation
import cv2
import PIL
from PIL import Image

import torch
from torch.utils.data import  Dataset
from torchvision import transforms

class RSNAMamographyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.df = pd.read_csv(annotations_file)
        # MODIFICATION. Drop all difficult negative casses
        self.df = self.df.drop(self.df[self.df['difficult_negative_case'] == 1].index)
        print(len(self.df))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    


    def __getitem__(self, ind):
        
        img_path = f"{self.img_dir}/{self.df.iloc[ind].patient_id}_{self.df.iloc[ind].image_id}.png"
        img = Image.open(img_path).convert('RGB')
        
        label = self.df.iloc[ind].cancer
        # there is no need to normalize data, it has already been normalized
        if self.transform:
            img = self.transform(img).to(torch.float32) 
        else:
            default_transform = transforms.Compose([transforms.ToTensor()])
            img = default_transform(img).to(torch.float32)
            
        #sample = {"image" : img, "label": label}
        return img, label