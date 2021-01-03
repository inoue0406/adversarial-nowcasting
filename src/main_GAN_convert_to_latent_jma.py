import numpy as np
import torch 
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import pandas as pd
import h5py
import os
import sys
import json
import time

from scaler import *
from train_valid_epoch_advloss import *
from utils import Logger
from opts import parse_opts
from loss_funcs import *
from loss_funcs_discriminator import *

device = torch.device("cuda")

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")

def root_scaling(X):
    a = 1/3.0    # "cubic root"
    return (X)**a

def to_GAN_input(input,gan_img_size):
    # upsample to 256x256 for GAN input
    input_GAN = F.interpolate(torch.squeeze(input),(gan_img_size,gan_img_size))
    input_GAN = root_scaling(input_GAN) # scaling for GAN
    input_GAN = torch.stack(3*[input_GAN],axis=2) # convert to 3 channels
    return input_GAN        
    
if __name__ == '__main__':
   
    # parse command-line options
    opt = parse_opts()
    print(opt)
    # create result dir
    if not os.path.exists(opt.result_path):
        os.mkdir(opt.result_path)
    
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    # generic log file
    logfile = open(os.path.join(opt.result_path, 'log_run.txt'),'w')
    logfile.write('Start time:'+time.ctime()+'\n')
    tstart = time.time()

    # model information
    modelinfo = open(os.path.join(opt.result_path, 'model_info.txt'),'w')

    # prepare scaler for data
    if opt.dataset == 'radarJMA' or opt.dataset == 'radarJMA3' :
        if opt.data_scaling == 'linear':
            scl = LinearScaler()
        elif opt.data_scaling == 'root':
            scl = RootScaler()
        elif opt.data_scaling == 'root_int':
            scl = RootIntScaler()
        elif opt.data_scaling == 'log':
             scl = LogScaler()
    elif opt.dataset == 'artfield':
        if opt.data_scaling == 'linear':
            # use identity transformation, since the data is already scaled
            scl = LinearScaler(rmax=1.0)

    # pretrained GAN model
    gan_img_size = 256
    GAN = LightweightGAN(
        optimizer="adam",
        lr = 2e-4,
        latent_dim = 256,
        attn_res_layers = [32],
        sle_spatial = False,
        image_size = gan_img_size,
        ttur_mult = 1.,
        fmap_max = 512,
        disc_output_size = 1,
        transparent = False,
        rank = 0
    )
    load_data = torch.load(opt.gan_path)
    print("number of GAN weights", len(load_data["GAN"].keys()))
    GAN.load_state_dict(load_data['GAN'])
    GAN.to(device)
    
    # freeze all GAN parameters
    for parameter in GAN.parameters():
        parameter.requires_grad = False
    
    if not opt.no_train:
        # prepare transform
        composed = None
        # loading datasets
        if opt.dataset == 'radarJMA':
            from jma_pytorch_dataset import *
            train_dataset = JMARadarDataset(root_dir=opt.data_path,
                                            csv_file=opt.train_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
            
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
#                                                   shuffle=True)
                                                   shuffle=False)
            
        if opt.loss_function == 'MSE':
            loss_fn = torch.nn.MSELoss()
            
        # training
        for i_batch, sample_batched in enumerate(train_loader):
            input = Variable(scl.fwd(sample_batched['past'].float())).cuda()
            target = Variable(scl.fwd(sample_batched['future'].float())).cuda()

            # concatenate along time axis
            in_tg =  torch.cat([input,target],axis=1)
            
            input_gan = to_GAN_input(in_tg,gan_img_size)

            # time series of latent variables
            latent_dim = 256
            latent_ts = torch.zeros(opt.batch_size, 2*opt.tdim_use, latent_dim).cuda()

            for b in range(opt.batch_size):
                print("batch:",b)
                # initialize latent vector
                # note that the latent vector itself is the target of optimization
                
                # random initialization
                tmp = torch.randn(latent_dim)
                latents = torch.stack(2*opt.tdim_use*[tmp],axis=0).cuda() # use same initial value for each time
                ## use previous latent as initial value
                #latents = latent_ts[:,t-1,:]

                latents.requires_grad=True

                # Type of optimizers adam/rmsprop
                if opt.optimizer == 'adam':
                    optimizer = torch.optim.Adam([latents], lr=opt.learning_rate)
                elif opt.optimizer == 'rmsprop':
                    optimizer = torch.optim.RMSprop([latents], lr=opt.learning_rate)
            
                # learning rate scheduler
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
                
                for n in range(opt.n_epochs):
                    # Generator
                    generated_images = GAN.G(latents)
                
                    # calc loss between generated & images
                    loss_mse = loss_fn(input_gan[b,:,:,:,:],generated_images)
                    # consecutive images should be continuous
                    loss_cont = loss_fn(latents[1:,:],latents[:-1,:])
                    loss = loss_mse + opt.loss_weights[0] * loss_cont
            
                    # Forward + Backward + Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("reconstruction loss =",n,loss_mse.data.cpu().numpy(),loss_cont.data.cpu().numpy())

                # save to h5 file
                h5path = opt.result_path + "/"+ sample_batched['fnames_past'][b]
                h5file = h5py.File(h5path,'w')
                h5file.create_dataset('latent',data= latents.data.cpu())
                h5file.close()  
                del latents

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
