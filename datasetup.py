import splitfolders
import conf as cfg
import fnmatch
import os
from matplotlib import pyplot as plt
import cv2
from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import *
from torchvision import transforms, utils, datasets, models
from PIL import Image
from pdb import set_trace
import time
import copy
from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import io, transform
from tqdm import trange, tqdm
import csv
import glob
# import dlib
import pandas as pd
import numpy as np


face_cascade = cv2.CascadeClassifier(cfg.paths['haar'])

def face_detection(paths: str):
    for root,_,files in os.walk(paths):
        for filename in files: 
            file = os.path.join(root,filename)
            if fnmatch.fnmatch(file,'*.jpg'):
                
                img = cv2.imread(file)        
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    crop_face = img[y:y+h, x:x+w]

                path = os.path.join(root,filename)
                cv2.imwrite(path,crop_face)




def split_data():
    splitfolders.ratio(cfg.paths['pre_dataset'], output=cfg.paths['data'], seed=1337, ratio=(.8, 0.2))




data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.NEAREST,expand=False, center=None),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
       transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.NEAREST,expand=False, center=None),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}
# data_dir = '/content/drive/MyDrive/AttendanceCapturingSystem/data/'
data_dir = os.path.join(cfg.paths['dataset'])
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=64, 
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)  # pause a bit so that plots are updated
# Get a batch of training data
    plt.show()




def main():
    print(2)
    # split_data()
    # face_detection(paths=cfg.paths['dataset'])
    # print(class_names)
    # inputs, classes = next(iter(dataloaders['train']))
    # # Make a grid from batch
    # out = utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])

    # batch = next(iter(dataloaders['train']))
    # print(batch[0].shape, batch[1].shape)
    # # print(dataloaders['train'])

if __name__ == '__main__':
    main()