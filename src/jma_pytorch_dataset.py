import torch 
import torchvision
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os

# Pytorch custom dataset for JMA Radar data
# The class assumes the data to be in h5 format

class JMARadarDataset(data.Dataset):
    def __init__(self,csv_file,root_dir,tdim_use=12,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the radar data.
            tdim_use: Size of temporal data to be used
                       ex) tdim_use=3 means last 3 of X and first 3 of Y are used
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df_fnames = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tdim_use = tdim_use
        self.transform = transform
        
    def __len__(self):
        return len(self.df_fnames)
        
    def __getitem__(self, index):
        # read X
        h5_name_X = os.path.join(self.root_dir, self.df_fnames.iloc[index].loc['fname'])
        h5file = h5py.File(h5_name_X,'r')
        rain_X = h5file['R'][()].astype(np.float32)
        rain_X = np.maximum(rain_X,0) # replace negative value with 0
        rain_X = rain_X[-self.tdim_use:,None,:,:] # add "channel" dimension as 1
        h5file.close()
        # read Y
        h5_name_Y = os.path.join(self.root_dir, self.df_fnames.iloc[index].loc['fnext'])
        h5file = h5py.File(h5_name_Y,'r')
        rain_Y = h5file['R'][()].astype(np.float32)
        rain_Y = np.maximum(rain_Y,0) # replace negative value with 0
        rain_Y = rain_Y[:self.tdim_use,None,:,:] # add "channel" dimension as 1
        h5file.close()
        # save
        fnames_past = self.df_fnames.iloc[index].loc['fname']
        fnames_future = self.df_fnames.iloc[index].loc['fnext']
        sample = {'past': rain_X, 'future': rain_Y,
                  'fnames_past':fnames_past,'fnames_future':fnames_future}
        #sample = {'past': rain_X, 'future': rain_Y}

        if self.transform:
            sample = self.transform(sample)

        return sample

class JMARadarDataset3(data.Dataset):
    def __init__(self,csv_file,root_dir,tdim_use=12,transform=None,randinit=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the radar data.
            tdim_use: Size of temporal data to be used
                       ex) tdim_use=3 means last 3 of X and first 3 of Y are used
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df_fnames = pd.read_csv(csv_file)
        self.df_fnames = self.df_fnames.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 
        self.root_dir = root_dir
        self.tdim_use = tdim_use
        self.transform = transform
        self.randinit = randinit
        
    def __len__(self):
        return len(self.df_fnames)
        
    def __getitem__(self, index):
        # Read 3 files
        
        # read file1
        h5_name_X = os.path.join(self.root_dir, self.df_fnames.iloc[index].loc['fname'])
        h5file = h5py.File(h5_name_X,'r')
        rain_X1 = h5file['R'][()]
        rain_X1 = np.maximum(rain_X1,0) # replace negative value with 0
        rain_X1 = rain_X1[:,None,:,:] # add "channel" dimension as 1
        h5file.close()
        
        # read file2
        h5_name_Y = os.path.join(self.root_dir, self.df_fnames.iloc[index].loc['fname1'])
        h5file = h5py.File(h5_name_Y,'r')
        rain_X2 = h5file['R'][()]
        rain_X2 = np.maximum(rain_X2,0) # replace negative value with 0
        rain_X2 = rain_X2[:,None,:,:] # add "channel" dimension as 1
        
        # read file3
        h5_name_Y = os.path.join(self.root_dir, self.df_fnames.iloc[index].loc['fname2'])
        h5file = h5py.File(h5_name_Y,'r')
        rain_X3 = h5file['R'][()]
        rain_X3 = np.maximum(rain_X3,0) # replace negative value with 0
        rain_X3 = rain_X3[:,None,:,:] # add "channel" dimension as 1
        h5file.close()

        # randomly select the beginning
        rain_all = np.concatenate([rain_X1,rain_X2,rain_X3])
        if self.randinit:
            n1 = np.random.choice(np.arange(0,rain_all.shape[0] - self.tdim_use*2))
        else:
            n1 = 0
        n2 = n1 + self.tdim_use
        n3 = n2 + self.tdim_use
        rain_X = rain_all[n1:n2,:,:,:]
        rain_Y = rain_all[n2:n3,:,:,:]

        # randomly select spatially
        dx = int(rain_X.shape[2]/2)
        if self.randinit:
            ix1 = np.random.choice(np.arange(0,dx))
            ix2 = ix1 + dx
            iy1 = np.random.choice(np.arange(0,dx))
            iy2 = iy1 + dx
            rain_X = rain_X[:,:,ix1:ix2,ix1:ix2]
            rain_Y = rain_Y[:,:,ix1:ix2,ix1:ix2]
        else:
            rain_X = rain_X[:,:,int(dx-dx/2):int(dx+dx/2),int(dx-dx/2):int(dx+dx/2)]
            rain_Y = rain_Y[:,:,int(dx-dx/2):int(dx+dx/2),int(dx-dx/2):int(dx+dx/2)]
            
        # save
        fnames_past = self.df_fnames.iloc[index].loc['fname']
        fnames_future1 = self.df_fnames.iloc[index].loc['fname1']
        fnames_future2 = self.df_fnames.iloc[index].loc['fname2']
        sample = {'past': rain_X, 'future': rain_Y,
                  'fnames_past':fnames_past,'fnames_future1':fnames_future1, 'fnames_future2':fnames_future2}

        if self.transform:
            sample = self.transform(sample)

        return sample
