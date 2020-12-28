# custom loss function for Pytorch

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

# GAN model
from LightweightGAN import LightweightGAN

def root_scaling(X):
    a = 1/3.0    # "cubic root"
    return (X)**a

class GAN_D_MSE_loss(nn.Module):
    # discriminator loss
    def __init__(self, GAN, weight, gan_img_size):
        super(GAN_D_MSE_loss, self).__init__()
        self.D = GAN.D
        self.weight = weight
        self.gan_img_size = gan_img_size
        self.mseloss = torch.nn.MSELoss()
        
    def GAN_discriminator(self,input):
        batch_size = input.shape[0]
        # force [0-1] range
        input = torch.clamp(input,min=0.0,max=1.0)
        # upsample to 256x256 for GAN input
        input_GAN = F.interpolate(torch.squeeze(input),(self.gan_img_size,self.gan_img_size))
        input_GAN = root_scaling(input_GAN) # scaling for GAN
        input_GAN = torch.stack(3*[input_GAN],axis=2) # convert to 3 channels
        disc = torch.zeros(input_GAN.shape[0:2]).cuda()
        for b in range(batch_size):
            disc_input, _ , _ = self.D(input_GAN[b,:,:,:,:])
            disc[b,:]=disc_input.squeeze()
        # sigmoid for avoiding negative values in discriminator
        return torch.sigmoid(disc)
    
    def forward(self, output, target):
        lmse = self.mseloss(output,target)
        #lgan = 1.0 * self.weight*torch.sum(1-self.GAN_discriminator(output))
        lgan = 1.0 * self.weight*torch.sum(self.GAN_discriminator(output))
        print("lmse,lgan=",lmse.data.cpu().numpy(),lgan.data.cpu().numpy())
        return lmse + lgan

