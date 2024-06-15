import pickle
import os 
from utils import same_seeds,cifar10_training_data 
from runners.improved_diffusion import Diffusion_ddim
import random
import pickle
import torchvision
import numpy as np
import argparse
import logging
import yaml
import torch
import io
import torch.nn as nn
import torch.nn.functional as F
import time
from PIL import Image
from torchvision import transforms
import utils
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='sparse data trained diffusion purification stage of ECLIPSE')
    parser.add_argument('--config', type=str, default='cifar10.yml', help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed') 
    parser.add_argument('--t', type=int, default=100, help='Sampling noise scale') 
    parser.add_argument('--sparse_diff_model', type=str, default="ema_0.9999_250000.pt")
    parser.add_argument('--sparse_set', type=str, default="test2000ps8000")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--poison', type=str, default="EM") 
    args = parser.parse_args()
    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)        
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device
    return args, new_config

class SDE_Diffusion(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args   
        self.runner = Diffusion_ddim(args, config, device=config.device) 
        self.register_buffer('counter', torch.zeros(1, device=config.device))
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)
    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')
        start_time = time.time()
        x_re = self.runner.diffusion_purification((x - 0.5) * 2)
        minutes, seconds = divmod(time.time() - start_time, 60)
        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))
        x_p = (x_re + 1) * 0.5
        self.counter += 1
        return x_p

if __name__ == '__main__':
    args, config = parse_args_and_config()
    same_seeds(args.seed) 

    """Adding Gaussian noise + denoising by sparse trained diffusion model"""
    model = SDE_Diffusion(args, config).eval().to(config.device)      

    poison_data_path = os.path.join("./poisoned_data/cifar10",args.poison+".pkl")
    poison_train_dataset = cifar10_training_data(poison_data_path, transform=transforms.Compose([transforms.ToTensor(),]))
    poison_train_loader = DataLoader(poison_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    iteration = args.sparse_diff_model.split("9_")[1].split(".pt")[0]
    purified_data_path = os.path.join("./purified_data/cifar10", args.sparse_set, str(args.t), iteration)
    if not os.path.exists(purified_data_path):
        os.makedirs(purified_data_path)
    all_purified_data = []

    print(f"[Poison:{args.poison}] [t*:{args.t}] [Sparse set:{args.sparse_set}] [Iteration:{iteration}]")

    """Purification stage: Adding Gaussian noise + denoising"""
    for batch, (poi_sample, label) in enumerate(poison_train_loader):
        print("Purification epoch: [{}/{}]".format(batch, int(len(poison_train_dataset)/args.batch_size)))
        purified_sample = model(poi_sample)

        for i in range(len(purified_sample)):
            img = purified_sample[i].cpu().clone()           
            img_pil = ToPILImage()(img)
            img_data = img_pil.tobytes()          
            all_purified_data.append((img_data, int(label[i])))
    
    with open(os.path.join(purified_data_path, args.poison + '-pure.pkl'), 'wb') as f:
        pickle.dump(all_purified_data, f)

 
  

