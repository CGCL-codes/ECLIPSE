import sys
import argparse
from typing import Any
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import os 
import dataset
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import os
import pickle
import io



class cifar10_training_data(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data, label = self.data[idx]
        img = Image.frombytes('RGB', (32, 32), img_data)
        img = img.convert('RGB')  # Convert image to RGB format 
        return self.transform(img), torch.tensor(label)
 
def add_random_gaussian_noise(tensor, std=0.1):
    noise = torch.randn_like(tensor) * std
    noisy_tensor = torch.clamp(tensor + noise, 0, 1)
    return noisy_tensor

def standard_loss(args, model, x, y):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, logits

def same_seeds(seed):
    torch.manual_seed(seed)   
    if torch.cuda.is_available():  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)   
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False   
    torch.backends.cudnn.deterministic = True   

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

 

 
 
 