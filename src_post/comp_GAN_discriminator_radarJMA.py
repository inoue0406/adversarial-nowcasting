#
# Plot Predicted Rainfall Data
#
import torch
import torchvision
import numpy as np
import torch.utils.data as data
from torch import nn, einsum
import torch.nn.functional as F

import pandas as pd
import h5py
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------
# add "src" as import path
path = os.path.join('../src')
sys.path.append(path)

from jma_pytorch_dataset import *
from scaler import *
from colormap_JMA import Colormap_JMA

# trajGRU model
from collections import OrderedDict
from models_trajGRU.network_params_trajGRU import model_structure_convLSTM, model_structure_trajGRU
from models_trajGRU.forecaster import Forecaster
from models_trajGRU.encoder import Encoder
from models_trajGRU.model import EF
from models_trajGRU.trajGRU import TrajGRU
from models_trajGRU.convLSTM import ConvLSTM
device = torch.device("cuda")

# GAN model
from LightweightGAN import LightweightGAN

def GAN_discriminator(input,D,gan_img_size,batch_size):
    # force [0-1] range
    input = torch.clamp(input,min=0.0,max=1.0)
    # upsample to 256x256 for GAN input
    input_GAN = F.interpolate(torch.squeeze(input),(gan_img_size,gan_img_size))
    input_GAN = root_scaling(input_GAN) # scaling for GAN
    input_GAN = torch.stack(3*[input_GAN],axis=2) # convert to 3 channels
    disc = torch.zeros(input_GAN.shape[0:2])
    for b in range(batch_size):
        disc_input, _ , _ = D(input_GAN[b,:,:,:,:].detach())
        disc[b,:]=disc_input.squeeze()
    return disc

def root_scaling(X):
    a = 1/3.0    # "cubic root"
    return (X)**a

def mod_str_interval(inte_str):
    # a tweak for decent filename 
    inte_str = inte_str.replace('(','')
    inte_str = inte_str.replace(']','')
    inte_str = inte_str.replace(',','')
    inte_str = inte_str.replace(' ','_')
    return(inte_str)

# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_path,filelist,model_name,model_fname,model_fname_gan,batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,mode='png_whole'):
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

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
    load_data = torch.load(model_fname_gan)
    
    print("number of GAN weights", len(load_data["GAN"].keys()))

    GAN.load_state_dict(load_data['GAN'])

    # Generator and Discriminator
    G = GAN.G
    D = GAN.D
    D_aug = GAN.D_aug

    #latent_dim = 256
    #latents = torch.randn(batch_size, latent_dim).cuda()
    #generated_images = G(latents)
    #fake_output, fake_output_32x32, _ = D(generated_images.detach())

    # dataset
    valid_dataset = JMARadarDataset(root_dir=data_path,
                                    csv_file=filelist,
                                    tdim_use=tdim_use,
                                    transform=None)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
 
    # load the saved model
    if model_name == 'clstm':
        # convolutional lstm
        from models_trajGRU.model import EF
        encoder_params,forecaster_params = model_structure_convLSTM(img_size,batch_size,model_name)
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1],tdim_use).to(device)
        model = EF(encoder, forecaster).to(device)
    elif model_name == 'trajgru':
        # trajGRU model
        from models_trajGRU.model import EF
        encoder_params,forecaster_params = model_structure_trajGRU(img_size,batch_size,model_name)
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1],tdim_use).to(device)
        model = EF(encoder, forecaster).to(device)
    # load weights
    model.load_state_dict(torch.load(model_fname))
    # evaluation mode
    model.eval()
    #
    for i_batch, sample_batched in enumerate(valid_loader):
        fnames = sample_batched['fnames_future']
        # skip if no overlap
        if len(set(fnames) &  set(df_sampled['fname'].values)) == 0:
            print("skipped batch:",i_batch)
            continue
        # apply the trained model to the data
        input = scl.fwd(sample_batched['past']).cuda()
        target = scl.fwd(sample_batched['future']).cuda()
        output = model(input)

        # apply GAN discriminator
        disc_input = GAN_discriminator(input,D,gan_img_size,batch_size)
        disc_target = GAN_discriminator(target,D,gan_img_size,batch_size)
        disc_output = GAN_discriminator(output,D,gan_img_size,batch_size)
        
        # Output only selected data in df_sampled
        for n,fname in enumerate(fnames):
            if (not (fname in df_sampled['fname'].values)):
                print('skipped:',fname)
                continue            
            vi = disc_input[0,:].cpu().data.numpy()
            vt = disc_target[0,:].cpu().data.numpy()
            vo = disc_output[0,:].cpu().data.numpy()
            df = pd.DataFrame({"time":(np.arange(24)+1)*5,
                               "truth":np.concatenate([vi,vt]),
                               "model":np.concatenate([vi,vo])})
            fname2 = fname.replace(".h5",".csv")
            df.to_csv(pic_path+'disc_output_'+fname2)
        # free GPU memory (* This seems to be necessary if going without backprop)
        del input,target,output
        
if __name__ == '__main__':
    # params
    batch_size = 4
    tdim_use = 12
    img_size = 200

    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    if argc != 3:
        print('Usage: python plot_comp_prediction.py CASENAME clstm/trajgru')
        quit()
        
    case = argvs[1]
    #case = 'result_20190625_clstm_lrdecay07_ep20'

    model_name = argvs[2]
    #model_name = 'clstm'
    #model_name = 'trajgru'

    # GAN data path
    case_gan = "result_GAN_201214_testrun_alljapan"
    model_fname_gan = case_gan + '/model_150.pt'
    #model_fname_gan = case_gan + '/model_1.pt'

    data_path = '../data/data_kanto/'
    filelist = '../data/valid_simple_JMARadar.csv'
    model_fname = case + '/trained_CLSTM.dict'
    pic_path = case + '/csv/'

    data_scaling = 'linear'
    
    # prepare scaler for data
    if data_scaling == 'linear':
        scl = LinearScaler()
    if data_scaling == 'root':
        scl = RootScaler()
    if data_scaling == 'root_int':
        scl = RootIntScaler()
    elif data_scaling == 'log':
        scl = LogScaler()

    # samples to be plotted
    sample_path = '../data/sampled_forplot_3day_JMARadar.csv'

    # read sampled data in csv
    df_sampled = pd.read_csv(sample_path)
    print('samples to be plotted')
    print(df_sampled)
    
    plot_comp_prediction(data_path,filelist,model_name,model_fname,model_fname_gan,
                         batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,mode='png_ind')


