import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.functional import normalize
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import csv

#usage
#   full_dataset = MultimodalDataset('./')
#   train_size = int(0.7 * len(full_dataset))
#   test_size = len(full_dataset) - train_size
#   train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

def norm(array):
    array_ = np.reshape(array, (-1, array.shape[-1]))
    mean = np.mean(array_)
    std = np.std(array_)
    return (array - mean)/std

class MultimodalDataset(Dataset):
    def __init__(
        self, data_dir, zero_padding = True):
        super(MultimodalDataset, self).__init__()

        self.data_dir = data_dir
        self.zero_padding = zero_padding
        csv_path_x = self.data_dir + 'datax_im3_timestep10_10.csv'
        csv_path_y = self.data_dir + 'datax_im3_timestep10_10.csv'

        with open(csv_path_x, 'r') as f:
            csv_reader = csv.reader(f)
            X = list(csv_reader)
        for i, j in enumerate(X):
            for m, n in enumerate(j):
                j[m] = float(n)
        with open(csv_path_y, 'r') as f:
            csv_reader = csv.reader(f)
            Y = list(csv_reader)
        for i, j in enumerate(Y):
            for m, n in enumerate(j):
                j[m] = float(n)
        datax = np.asarray(X)
        datay = np.asarray(Y)
        #datax shape (565780, 55)

        datax = np.reshape(datax,(datay.shape[0], 10, 55))
        datay = np.reshape(datax,(datay.shape[0], 10, 2))
        datax = datax[:-10,:,:] #past information
        datay = datay[:-10,:] #future goal
        datax_p = datax[10:,:,:] #futrue information

        pose = datax[:,:,0:50]
        motion = datax[:,:,50:52]
        head = datax[:,:,52:55]
        motion_p = datax_p[:,:,50:52] #future motion
        #if you need future pose and head orientation, add them here.
        
        self.num_seq = datax.shape[0]

        #normalization
        pose = norm(pose)
        motion = norm(motion)
        head = norm(head)
        motion_p = norm(motion_p)
        datay = norm(datay)
        

        #convert numpy to tensor
        self.pose = torch.from_numpy(pose).type(torch.double)
        self.motion= torch.from_numpy(motion).type(torch.double)
        self.head = torch.from_numpy(head).type(torch.double)
        self.motion_p = torch.from_numpy(motion_p).type(torch.double)
        self.datay = torch.from_numpy(datay).type(torch.double)

    
    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, index):
        out = [
            self.motion[index], self.motion_p[index],
            self.pose[index], self.head[index], 
            self.datay[index]
        ]
        return out