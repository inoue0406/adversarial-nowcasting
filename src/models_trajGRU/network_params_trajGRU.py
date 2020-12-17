from collections import OrderedDict

from models_trajGRU.trajGRU import TrajGRU
from models_trajGRU.convLSTM import ConvLSTM

from models_trajGRU.model import activation
ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)

def model_structure_convLSTM(image_size,batch_size,model_name):
    # model structure
    # build model for convlstm

    if model_name == "clstm":
        # for normal spatio-temporal prediction, set 1
        num_last_layer = 1
    elif model_name == "clstm_el":
        # for Euler-Lagrange model, set 3
        num_last_layer = 3
    
    if image_size == 128:
        convlstm_encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 8, 7, 3, 1]}),
                OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
                OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
            ],
    
            [
                ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 42, 42),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 14, 14),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 7, 7),
                         kernel_size=3, stride=1, padding=1),
            ]
        ]
        convlstm_forecaster_params = [
            [
                OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
                OrderedDict({
                    'deconv3_leaky_1': [64, 8, 7, 3, 1],
                    'conv3_leaky_2': [8, 8, 3, 1, 1],
                    'conv3_3': [8, num_last_layer, 1, 1, 0]
                }),
            ],
    
            [
                ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 7, 7),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 14, 14),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 42, 42),
                         kernel_size=3, stride=1, padding=1),
            ]
        ]
    if image_size == 200:
        convlstm_encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 8, 7, 3, 1]}),
                OrderedDict({'conv2_leaky_1': [64, 128, 5, 3, 1]}),
                OrderedDict({'conv3_leaky_1': [128, 128, 3, 2, 1]}),
            ],
    
            [
                ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 66, 66),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=128, num_filter=128, b_h_w=(batch_size, 22, 22),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=128, num_filter=128, b_h_w=(batch_size, 11, 11),
                         kernel_size=3, stride=1, padding=1),
            ]
        ]
        convlstm_forecaster_params = [
            [
                OrderedDict({'deconv1_leaky_1': [128, 128, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [128, 64, 5, 3, 1]}),
                OrderedDict({
                    'deconv3_leaky_1': [64, 8, 7, 3, 1],
                    'conv3_leaky_2': [8, 8, 3, 1, 1],
                    'conv3_3': [8, num_last_layer, 1, 1, 0]
                }),
            ],
    
            [
                ConvLSTM(input_channel=128, num_filter=128, b_h_w=(batch_size, 11, 11),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=128, num_filter=128, b_h_w=(batch_size, 22, 22),
                         kernel_size=3, stride=1, padding=1),
                ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 66, 66),
                         kernel_size=3, stride=1, padding=1),
            ]
        ]
        
    return convlstm_encoder_params,convlstm_forecaster_params
          
def model_structure_trajGRU(image_size,batch_size,model_name):
    # model structure
    # parameters for trajGRU

    if model_name == "trajgru":
        # for normal spatio-temporal prediction, set 1
        num_last_layer = 1
    elif model_name == "trajgru_el":
        # for Euler-Lagrange model, set 3
        num_last_layer = 3
    
    if image_size == 128:
        trajgru_encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 8, 7, 3, 1]}),
                OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
                OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
            ],
            
            [
                TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 42, 42), zoneout=0.0, L=13,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
                
                TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 14, 14), zoneout=0.0, L=13,
         
               i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
                TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 7, 7), zoneout=0.0, L=9,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE)
            ]
        ]
        trajgru_forecaster_params = [
            [
                OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
                OrderedDict({
                    'deconv3_leaky_1': [64, 8, 7, 3, 1],
                    'conv3_leaky_2': [8, 8, 3, 1, 1],
                    'conv3_3': [8, num_last_layer, 1, 1, 0]
                }),
            ],
        
            [
                TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 7, 7), zoneout=0.0, L=13,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
        
                TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 14, 14), zoneout=0.0, L=13,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
                TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 42, 42), zoneout=0.0, L=9,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE)
            ]
        ]
    elif image_size == 200:
        trajgru_encoder_params = [
            [
                OrderedDict({'conv1_leaky_1': [1, 8, 7, 3, 1]}),
                OrderedDict({'conv2_leaky_1': [32, 96, 5, 3, 1]}),
                OrderedDict({'conv3_leaky_1': [96, 96, 3, 2, 1]}),
            ],
            
            [
                TrajGRU(input_channel=8, num_filter=32, b_h_w=(batch_size, 66,66), zoneout=0.0, L=13,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
                
                TrajGRU(input_channel=96, num_filter=96, b_h_w=(batch_size, 22, 22), zoneout=0.0, L=13,
         
               i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
                TrajGRU(input_channel=96, num_filter=96, b_h_w=(batch_size, 11, 11), zoneout=0.0, L=9,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE)
            ]
        ]
        trajgru_forecaster_params = [
            [
                OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
                OrderedDict({'deconv2_leaky_1': [96, 32, 5, 3, 1]}),
                OrderedDict({
                    'deconv3_leaky_1': [32, 8, 7, 3, 1],
                    'conv3_leaky_2': [8, 8, 3, 1, 1],
                    'conv3_3': [8, num_last_layer, 1, 1, 0]
                }),
            ],
        
            [
                TrajGRU(input_channel=96, num_filter=96, b_h_w=(batch_size, 11, 11), zoneout=0.0, L=13,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
        
                TrajGRU(input_channel=96, num_filter=96, b_h_w=(batch_size, 22, 22), zoneout=0.0, L=13,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE),
                TrajGRU(input_channel=32, num_filter=32, b_h_w=(batch_size, 66, 66), zoneout=0.0, L=9,
                        i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                        h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                        act_type=ACT_TYPE)
            ]
        ]
        
    return trajgru_encoder_params,trajgru_forecaster_params
