import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import pandas as pd
import h5py
import os
import re

# Pytorch custom dataset for JMA timeseries data

class JMATSDataset(data.Dataset):
    def __init__(self,csv_data,root_dir,tdim_use=12,transform=None):
        """
        Args:
            csv_data (string): Path to the csv file with time series data.
            root_dir (string): Directory with all the radar data.
            tdim_use: Size of temporal data to be used
                       ex) tdim_use=3 means last 3 of X and first 3 of Y are used
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # read time series data
        self.df_fnames = pd.read_csv(csv_data)
        print("data length of original csv",len(self.df_fnames))
        
        self.root_dir = root_dir
        self.tdim_use = tdim_use
        self.transform = transform
        
    def __len__(self):
        return len(self.df_fnames)
        
    def __getitem__(self, index):
        # read time series data
        h5_name_X = os.path.join(self.root_dir, self.df_fnames.iloc[index].loc['fname'])
        h5file = h5py.File(h5_name_X,'r')
        rain_X = h5file['latent'][()].astype(np.float32)
        # set past-future pair
        rain_past = rain_X[0:self.tdim_use,:]
        rain_future = rain_X[self.tdim_use:2*self.tdim_use,:]
                
        sample = {'past': rain_past,
                  'future': rain_future}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
