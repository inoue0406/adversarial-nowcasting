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

import datetime

# Pytorch custom dataset for JMA timeseries data

def fname_1h_later(fname):
    f2 = fname.replace("2p-jmaradar5_","").replace("utc.h5","")
    dt = datetime.datetime.strptime(f2,'%Y-%m-%d_%H%M')
    # +1h data for Y
    date1 = dt + pd.offsets.Hour()
    fname1 = date1.strftime('2p-jmaradar5_%Y-%m-%d_%H%Mutc.h5')
    return fname1

class JMATSDataset(data.Dataset):
    def __init__(self,csv_data,root_dir,root_dir_grid,tdim_use=12,transform=None):
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
        
        if root_dir_grid != None:
            self.mode = "test"
            self.root_dir_grid = root_dir_grid
        else:
            self.mode = "train"
        
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

        if self.mode == "train":
            # if train mode, only use latent variables
            sample = {'past': rain_past,
                      'future': rain_future}
        elif self.mode == "test":
            # if test mode, load gridded data as well
            # read X
            fname_X = self.df_fnames.iloc[index].loc['fname']
            h5_name_X = os.path.join(self.root_dir_grid, fname_X)
            h5file = h5py.File(h5_name_X,'r')
            rain_X = h5file['R'][()].astype(np.float32)
            rain_X = np.maximum(rain_X,0) # replace negative value with 0
            rain_X = rain_X[-self.tdim_use:,None,:,:] # add "channel" dimension as 1
            h5file.close()
            # read Y
            fname_Y = fname_1h_later(fname_X)
            h5_name_Y = os.path.join(self.root_dir_grid, fname_Y)
            h5file = h5py.File(h5_name_Y,'r')
            rain_Y = h5file['R'][()].astype(np.float32)
            rain_Y = np.maximum(rain_Y,0) # replace negative value with 0
            rain_Y = rain_Y[:self.tdim_use,None,:,:] # add "channel" dimension as 1
            h5file.close()
            
            sample = {'past': rain_past,
                      'future': rain_future,
                      'past_grid': rain_X,
                      'future_grid': rain_Y,
                      'fnames_past':h5_name_X,'fnames_future':h5_name_Y}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
