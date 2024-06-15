import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
from utils import same_seeds, standard_loss, cifar10_training_data, add_random_gaussian_noise 
import time
import warnings
import argparse
import numpy as np
import csv
from torch.autograd import Variable 
from classifiers.resnet import resnet18, resnet50
from classifiers.vgg import VGG16, VGG19
from classifiers.densenet import DenseNet121 



warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Training on poisoned data w/ or w/o ECLIPSE: Testing the effectiveness of ECLIPSE')
parser.add_argument('--lr', default=0.1, type=float, help='learning-rate')
parser.add_argument('--epochs', default=80, type=int, help='number of epoch') 
parser.add_argument('--arch', default='resnet18', type=str, help='types of training architecture')   
parser.add_argument('--poison', default="EM", type=str, help="EM, TAP, CUDA, ...")
parser.add_argument('--batch_size', default=128, type=int)  
parser.add_argument('--t', default=100, type=int)
parser.add_argument('--iter', default=250000, type=int)
parser.add_argument('--seed', default=0, type=int)   
parser.add_argument('--std', default=0.05, type=float, help="std of gaussian noise")
parser.add_argument('--p', default=0.4, type=float, help="grayscale transformation probability")
parser.add_argument('--sparse_set', default='test2000ps8000', type=str)
parser.add_argument('--pure', action='store_true', help="whether to employ ECLIPSE defense")
args = parser.parse_args() 

 
same_seeds(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
 
 

if args.pure:         #applying lightweight corruption compensation module of ECLIPSE
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=args.p),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_random_gaussian_noise(x, std=args.std)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])    
 

else:        #applying standard transformation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
 

if args.pure:
    training_data_path = os.path.join("./purified_data/cifar10", args.sparse_set, str(args.t), str(args.iter), args.poison + '-pure.pkl')     #purified training data by ECLIPSE
else:        
    training_data_path = os.path.join("./poisoned_data/cifar10", args.poison + '.pkl')     #poisoned training data


 
train_dataset = cifar10_training_data(training_data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True) 

test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

 

if args.arch == "resnet18":
    net = resnet18(in_dims=3, out_dims=num_classes)
elif args.arch == "resnet50":
    net = resnet50(in_dims=3, out_dims=num_classes)
elif args.arch == "vgg16":
    net = VGG16()
elif args.arch == "vgg19":
    net = VGG19()
elif args.arch == "densenet121":
    net = DenseNet121() 
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)


print(f"[Pure:{args.pure}] [Arch:{args.arch}] [Poison:{args.poison}] [t*:{args.t}]\n[sparse_set:{args.sparse_set}] [Grayscale p:{args.p}] [Gaussian std:{args.std}]")
for epoch in range(args.epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    net.train()
    for i, (inputs, labels) in enumerate(train_loader, 0):      #index starts from 0
        inputs = torch.clamp(inputs, 0, 1)
        labels = labels.long()
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss, _ = standard_loss(args, net, inputs, labels)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        loss.backward()
        optimizer.step()

    print('[Epoch：%d/%d] loss: %.3f Train Acc: %.3f' % (epoch + 1, args.epochs, running_loss / len(train_loader), 100. * correct / total)) 
    running_loss = 0.0

    if (epoch + 1) % 5 == 0: 
        net.eval()
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(test_loader, 0):
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print('Test Acc: %.2f' % (100. * correct / total)) 

 
with open(os.path.join(f'results.csv'), 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([args.pure, args.arch, args.poison, args.sparse_set, args.t, args.p, args.std, 100 * correct / total])  
print('Finished Training')



