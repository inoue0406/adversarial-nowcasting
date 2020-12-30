import torch
from torch.autograd import Variable
import torchvision
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F

from jma_pytorch_dataset import *
from utils import AverageMeter, Logger
from criteria_precip import *
# for debug
from tools_mem import *

# training/validation for one epoch

# --------------------------
# Training
# --------------------------

def train_epoch(epoch,num_epochs,train_loader,model,loss_fn,optimizer,train_logger,train_batch_logger,opt):
    
    print('train at epoch {}'.format(epoch))
    # trainning mode
    model.train()

    losses = AverageMeter()
    
    for i_batch, sample_batched in enumerate(train_loader):
        #print(i_batch, sample_batched['past'].size(),sample_batched['future'].size())
        input = Variable(sample_batched['past'].float()).cuda()
        target = Variable(sample_batched['future'].float()).cuda()
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # for logging
        losses.update(loss.item(), input.size(0))

        print('chk lr ',optimizer.param_groups[0]['lr'])
        train_batch_logger.log({
            'epoch': epoch,
            'batch': i_batch+1,
            'loss': losses.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        if (i_batch+1) % 1 == 0:
            print ('Train Epoch [%d/%d], Iter [%d/%d] Loss: %.4e' 
                   %(epoch, num_epochs, i_batch+1, len(train_loader.dataset)//train_loader.batch_size, loss.item()))

    # update lr for optimizer
    optimizer.step()

    train_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    # free gpu memory
    del input,target,output,loss

# --------------------------
# Test
# --------------------------

def root_scaling_inv(X):
    a = 1/3.0    # "cubic root"
    return (X)**(1/a)

def test_epoch(test_loader,model,GAN,loss_fn,scl,opt):
    print('Testing for the model')

    size_org = 200
    
    # initialize
    RMSE_all = np.empty((0,opt.tdim_use),float)
    Xpast_all = np.empty((0,opt.tdim_use),float)
    Xtrue_all = np.empty((0,opt.tdim_use),float)
    Xmodel_all = np.empty((0,opt.tdim_use),float)

    # evaluation mode
    model.eval()

    for i_batch, sample_batched in enumerate(test_loader):
        past = sample_batched['past'].float()
        input = Variable(sample_batched['past'].float()).cuda()
        target = Variable(sample_batched['future'].float()).cuda()
        # Forward
        output = model(input)
        loss = loss_fn(output, target)
        
        # apply GAN to generate images
        
        for t in range(opt.tdim_use):
            output_images = GAN.G(output[:,t,:])
            output_images = F.interpolate(output_images,(size_org,size_org))
            output_images = root_scaling_inv(output_images)
            output_images = torch.mean(output_images,axis=1) # crush channel axis
            import pdb;pdb.set_trace()
        
        # concat all prediction data
        Xtrue = scl.inv(target.data.cpu().numpy().squeeze())
        Xmodel = scl.inv(output.data.cpu().numpy().squeeze())
        #
        Xpast_all = np.append(Xpast_all,past.squeeze(),axis=0)
        Xtrue_all = np.append(Xtrue_all,Xtrue,axis=0)
        Xmodel_all = np.append(Xmodel_all,Xmodel,axis=0)
        
        #if (i_batch+1) % 100 == 0:
        if (i_batch+1) % 1 == 0:
            print ('Test Iter [%d/%d] Loss: %.4e' 
                   %(i_batch+1, len(test_loader.dataset)//test_loader.batch_size, loss.item()))

    # prep csv
    tpred = (np.arange(opt.tdim_use)+1.0)*5.0 # in minutes
    RMSE = np.sqrt(np.mean((Xtrue_all-Xmodel_all)**2,axis=0))
    df_out = pd.DataFrame({'tpred_min':tpred,
                       'RMSE':RMSE})
    
    # apply eval metric by rain level
    levels = np.arange(-10,220,10)
    for i in range(len(levels)-1):
        low = levels[i]
        high = levels[i+1]
        id_range = (Xpast_all[:,-1] > low) * (Xpast_all[:,-1] <= high)
        print("range: ",low,high,"number of samples: ",np.sum(id_range))
        # calc rmse
        xt = Xtrue_all[id_range,:]
        xm = Xmodel_all[id_range,:]
        # RMSE along "samples" axis and keep time dim
        rr = np.sqrt(np.mean((xt-xm)**2,axis=0))
        vname = "RMSE_%d_%d" % (low,high)
        df_out[vname] = rr

    # save evaluated metric as csv file
    df_out.to_csv(os.path.join(opt.result_path,
                           'test_evaluation_predtime.csv'), float_format='%.3f')
    # free gpu memory
    del input,target,output,loss

    
