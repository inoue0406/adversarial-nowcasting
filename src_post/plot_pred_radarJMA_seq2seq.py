#
# Plot Predicted Rainfall Data
#
import torch
import torchvision
import numpy as np
import torch.utils.data as data

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

from jma_timeseries_dataset import *
from scaler import *
from train_valid_epoch_ts import *
from colormap_JMA import Colormap_JMA

# GAN model
from LightweightGAN import LightweightGAN

# seq2seq model
from models.seq2seq_lstm_ts import *
device = torch.device("cuda")

def mod_str_interval(inte_str):
    # a tweak for decent filename 
    inte_str = inte_str.replace('(','')
    inte_str = inte_str.replace(']','')
    inte_str = inte_str.replace(',','')
    inte_str = inte_str.replace(' ','_')
    return(inte_str)


def plot_pair_rainfall(pic_tg,pic_pred,fname,case,df_sampled,pf):
    # plot pairs of true/predicted field
    print('Plotting: ',fname,np.max(pic_tg),np.max(pic_pred))
    # plot
    cm = Colormap_JMA()
    for nt in range(6):
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.suptitle("Precip prediction starting at: "+fname+"\n"+case, fontsize=10)
        #        
        id = nt*2+1
        pos = nt+1
        dtstr = str((id+1)*5)
        # target
        plt.subplot(1,2,1)
        im = plt.imshow(pic_tg[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
        plt.title("true:"+pf+" "+dtstr+"min")
        plt.grid()
        # predicted
        plt.subplot(1,2,2)
        im = plt.imshow(pic_pred[id,:,:],vmin=0,vmax=50,cmap=cm,origin='lower')
        plt.title("pred:"+pf+" "+dtstr+"min")
        plt.grid()
        # color bar
        fig.subplots_adjust(right=0.93,top=0.85)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        # save as png
        i = df_sampled.index[df_sampled['fname']==fname]
        i = int(i.values)
        interval = mod_str_interval(df_sampled['rcategory'].iloc[i])
        nt_str = '_dt%02d' % nt
        plt.savefig(pic_path+'comp_pred_'+interval+fname+pf+nt_str+'.png')
        plt.close()

# plot comparison of predicted vs ground truth
def plot_comp_prediction(data_latent,data_grid,filelist,model_name,model_fname,
                         gan_path,batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,mode='png_whole'):
    # create pic save dir
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    # dataset
    test_dataset = JMATSDataset(csv_data=filelist,
                                root_dir=data_latent,
                                root_dir_grid=data_grid,
                                tdim_use=tdim_use,
                                transform=None)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              drop_last=True,
                                              shuffle=False)
    
    # load the saved model
    #load pretrained model from results directory
    print('loading pretrained model:',model_fname)
    model = torch.load(model_fname)
    loss_fn = torch.nn.MSELoss()

    # load GAN for spatial forecasting
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
    load_data = torch.load(gan_path)
    print("number of GAN weights", len(load_data["GAN"].keys()))
    GAN.load_state_dict(load_data['GAN'])
    GAN.cuda()
    # freeze all GAN parameters
    for parameter in GAN.parameters():
        parameter.requires_grad = False

    # evaluation mode
    model.eval()
    #
    for i_batch, sample_batched in enumerate(test_loader):
        fnames = sample_batched['fnames_future']
        # skip if no overlap
        if len(set(fnames) &  set(df_sampled['fname'].values)) == 0:
            print("skipped batch:",i_batch)
            continue
        
        # apply the trained model to the data
        input = Variable(sample_batched['past'].float()).cuda()
        target = Variable(sample_batched['future'].float()).cuda()
        # Forward
        output = model(input)

        # apply GAN to generate images
        size_org = img_size
        
        input_pred = np.zeros((batch_size,tdim_use,1,size_org,size_org))
        output_pred = np.zeros((batch_size,tdim_use,1,size_org,size_org))
        
        for t in range(tdim_use):
            # input
            input_images = GAN.G(input[:,t,:])
            input_images = F.interpolate(input_images,(size_org,size_org))
            input_images = root_scaling_inv(input_images)
            input_images = torch.mean(input_images,axis=1) # crush channel axis
            input_pred[:,t,0,:,:] = input_images.data.cpu().numpy()
            # output
            output_images = GAN.G(output[:,t,:])
            output_images = F.interpolate(output_images,(size_org,size_org))
            output_images = root_scaling_inv(output_images)
            output_images = torch.mean(output_images,axis=1) # crush channel axis
            output_pred[:,t,0,:,:] = output_images.data.cpu().numpy()
        
        # apply evaluation metric
        Xtrue_p = sample_batched['past_grid'].float().data.cpu().numpy()
        Xtrue_f = sample_batched['future_grid'].float().data.cpu().numpy()
        Xmodel_p = scl.inv(input_pred)
        Xmodel_f = scl.inv(output_pred)
                
        # Output only selected data in df_sampled
        for n,fname in enumerate(fnames):
            if (not (fname in df_sampled['fname'].values)):
                print('skipped:',fname)
                continue
            # do the plotting
            plot_pair_rainfall(Xtrue_p[n,:,0,:,:],Xmodel_p[n,:,0,:,:],
                                fname,case,df_sampled,"0past")
            plot_pair_rainfall(Xtrue_f[n,:,0,:,:],Xmodel_f[n,:,0,:,:],
                                fname,case,df_sampled,"1pred")
        # free GPU memory (* This seems to be necessary if going without backprop)
        del input,target,output
        
if __name__ == '__main__':
    # params
    batch_size = 4
    tdim_use = 12
    #img_size = 128
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

    data_latent = "result_20201229_gan_convert_to_latent_kanto_valid/"
    data_grid = "../data/data_kanto/"
    filelist = '../data/gan_latent_kanto_2017_full.csv'
    model_fname = case + '/trained_seq2seq.model'
    pic_path = case + '/png/'

    # pretrained GAN path
    gan_path = "../run/result_GAN_201214_testrun_alljapan/model_150.pt"
    
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
    
    plot_comp_prediction(data_latent,data_grid,filelist,model_name,model_fname,
                         gan_path,batch_size,tdim_use,
                         df_sampled,pic_path,scl,case,img_size,mode='png_ind')


