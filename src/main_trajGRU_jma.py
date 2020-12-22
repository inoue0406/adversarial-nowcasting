import numpy as np
import torch 
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import pandas as pd
import h5py
import os
import sys
import json
import time

from scaler import *
from train_valid_epoch import *
from utils import Logger
from opts import parse_opts
from loss_funcs import *

# trajGRU model
from collections import OrderedDict
from models_trajGRU.network_params_trajGRU import model_structure_convLSTM, model_structure_trajGRU
from models_trajGRU.forecaster import Forecaster
from models_trajGRU.encoder import Encoder
from models_trajGRU.trajGRU import TrajGRU
from models_trajGRU.convLSTM import ConvLSTM
from models_trajGRU.loss import Weighted_mse_mae
device = torch.device("cuda")

# data augmentation
from video_augmentation import RandomRotateVideo, RandomResizeVideo

def count_parameters(model,f):
    for name,p in model.named_parameters():
        f.write("name,"+name+", Trainable, "+str(p.requires_grad)+",#params, "+str(p.numel())+"\n")
    Nparam = sum(p.numel() for p in model.parameters())
    Ntrain = sum(p.numel() for p in model.parameters() if p.requires_grad)
    f.write("Number of params:"+str(Nparam)+", Trainable parameters:"+str(Ntrain)+"\n")
    
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


    if opt.model_name == 'clstm':
        # convolutional lstm
        from models_trajGRU.model import EF
        encoder_params,forecaster_params = model_structure_convLSTM(opt.image_size,opt.batch_size,opt.model_name)
        encoder = Encoder(encoder_params[0], encoder_params[1])
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1],opt.tdim_use)
        model = EF(encoder, forecaster)
    elif opt.model_name == 'trajgru':
        # trajGRU model
        from models_trajGRU.model import EF
        encoder_params,forecaster_params = model_structure_trajGRU(opt.image_size,opt.batch_size,opt.model_name)
        encoder = Encoder(encoder_params[0], encoder_params[1])
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1],opt.tdim_use)
        model = EF(encoder, forecaster)
    elif opt.model_name == 'trajgru_el':
        # trajGRU Euler-Lagrange Model
        from models_trajGRU.model_euler_lagrange import EF_el
        encoder_params,forecaster_params = model_structure_trajGRU(opt.image_size,opt.batch_size,opt.model_name)
        encoder = Encoder(encoder_params[0], encoder_params[1])
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1],opt.tdim_use)
        model = EF_el(encoder, forecaster, opt.image_size, opt.pc_size, opt.batch_size, opt.model_mode, opt.interp_type).to(device)

    # Data Parallel Multi-GPU Run
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model) # make parallel
    model.to(device)
    
    if not opt.no_train:
        # prepare transform
        if opt.aug_rotate > 0.0:
            Rot = RandomRotateVideo(degrees=opt.aug_rotate)
            Resize = RandomResizeVideo(factor=opt.aug_resize)
            composed = transforms.Compose([Rot,Resize])
        else:
            composed = None
        # loading datasets
        if opt.dataset == 'radarJMA':
            from jma_pytorch_dataset import *
            train_dataset = JMARadarDataset(root_dir=opt.data_path,
                                            csv_file=opt.train_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
            
            valid_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                            csv_file=opt.valid_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
        if opt.dataset == 'radarJMA3':
            from jma_pytorch_dataset import *
            train_dataset = JMARadarDataset3(root_dir=opt.data_path,
                                             csv_file=opt.train_path,
                                             tdim_use=opt.tdim_use,
                                             randinit=True,
                                             transform=composed)
            
            valid_dataset = JMARadarDataset3(root_dir=opt.valid_data_path,
                                            csv_file=opt.valid_path,
                                            tdim_use=opt.tdim_use,
                                            randinit=True,
                                            transform=None)
        elif opt.dataset == 'artfield':
            from artfield_pytorch_dataset import *
            train_dataset = ArtfieldDataset(root_dir=opt.data_path,
                                            csv_file=opt.train_path,
                                            mode=opt.model_mode,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
            
            valid_dataset = ArtfieldDataset(root_dir=opt.valid_data_path,
                                            csv_file=opt.valid_path,
                                            mode=opt.model_mode,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
    
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=True)
#                                                   shuffle=False)
    
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=opt.batch_size,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=False)
        
        #dd = next(iter(train_dataset))
    
        if opt.transfer_path != 'None':
            # Use pretrained weights for transfer learning
            print('loading pretrained model:',opt.transfer_path)
            # ###if the model is pickle
            #model = torch.load(opt.transfer_path)
            # ###if the model is state dict
            model.load_state_dict(torch.load(opt.transfer_path))

            model.model_mode = opt.model_mode

        modelinfo.write('Model Structure \n')
        modelinfo.write(str(model))
        count_parameters(model,modelinfo)
        modelinfo.close()
        
        if opt.loss_function == 'MSE':
            loss_fn = torch.nn.MSELoss()
        elif opt.loss_function == 'Weighted_MSE_MAE':
            loss_fn = Weighted_mse_mae(LAMBDA=0.01).to(device)
        elif opt.loss_function == 'WeightedMSE':
            loss_fn = weighted_MSE_loss(opt.loss_weights)
        elif opt.loss_function == 'MaxMSE':
            loss_fn = max_MSE_loss(opt.loss_weights)
        elif opt.loss_function == 'MultiMSE':
            loss_fn = multi_MSE_loss(opt.loss_weights)

        # Type of optimizers adam/rmsprop
        if opt.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        elif opt.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=opt.learning_rate)
            
        # learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.lr_decay)
            
        # Prep logger
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'loss', 'lr'])
        valid_logger = Logger(
            os.path.join(opt.result_path, 'valid.log'),
            ['epoch', 'loss'])
    
        # training 
        for epoch in range(1,opt.n_epochs+1):
            # step scheduler
            scheduler.step()
            # training & validation
            train_epoch(epoch,opt.n_epochs,train_loader,model,loss_fn,optimizer,
                        train_logger,train_batch_logger,opt,scl)
            #valid_epoch(epoch,opt.n_epochs,valid_loader,model,loss_fn,
            #            valid_logger,opt,scl)

            if epoch % opt.checkpoint == 0:
                # save the trained model for every checkpoint
                # (1) as binary 
                #torch.save(model,os.path.join(opt.result_path,
                #                                 'trained_CLSTM_epoch%03d.model' % epoch))
                # (2) as state dictionary
                torch.save(model.state_dict(),
                           os.path.join(opt.result_path,
                                        'trained_CLSTM_epoch%03d.dict' % epoch))
        # save the trained model
        # (1) as binary 
        #torch.save(model,os.path.join(opt.result_path, 'trained_CLSTM.model'))
        # (2) as state dictionary
        torch.save(model.state_dict(),
                   os.path.join(opt.result_path, 'trained_CLSTM.dict'))

    # test datasets if specified
    batch_size_test = 4
    if opt.test:
        if opt.no_train:
            #load pretrained model from results directory
            model_fname = os.path.join(opt.result_path, 'trained_CLSTM.dict')
            print('loading pretrained model:',model_fname)
            # ###if the model is pickle
            #model_ld = torch.load(model_fname)
            # ###if the model is state dict
            model.load_state_dict(torch.load(model_fname))
            # tweak
            #from models_trajGRU.model_euler_lagrange import EF_el
            #model = EF_el(model.encoder, model.forecaster,
            #              opt.image_size, opt.pc_size, batch_size_test, opt.model_mode, opt.interp_type).to(device)
            #del model_ld
            loss_fn = torch.nn.MSELoss()

        # smaller batch size is used, since trajGRU is heavy on memory
        #batch_size_test = 4
        #batch_size_test = opt.batch_size
        # prepare loader
        if opt.dataset == 'radarJMA':
            from jma_pytorch_dataset import *
            test_dataset = JMARadarDataset(root_dir=opt.valid_data_path,
                                            csv_file=opt.test_path,
                                            tdim_use=opt.tdim_use,
                                            transform=None)
        if opt.dataset == 'radarJMA3':
            from jma_pytorch_dataset import *
            test_dataset = JMARadarDataset3(root_dir=opt.valid_data_path,
                                            csv_file=opt.test_path,
                                            tdim_use=opt.tdim_use,
                                            randinit=False,
                                            transform=None)
                        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size_test,
                                                   num_workers=opt.n_threads,
                                                   drop_last=True,
                                                   shuffle=False)
        
        # testing for the trained model
        for threshold in opt.eval_threshold:
            test_CLSTM_EP(test_loader,model,loss_fn,opt,scl,threshold)

    # output elapsed time
    logfile.write('End time: '+time.ctime()+'\n')
    tend = time.time()
    tdiff = float(tend-tstart)/3600.0
    logfile.write('Elapsed time[hours]: %f \n' % tdiff)
