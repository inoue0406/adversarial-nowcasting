#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

case="result_20210102_gan_convert_latent_wt10_1000_kanto_valid"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_GAN_convert_to_latent_jma.py \
       --dataset radarJMA --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200\
       --train_path ../data/valid_simple_JMARadar_tmp_small.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.1 --lr_decay 0.9\
       --batch_size 4 --n_epochs 1000 --n_threads 4 \
       --loss_function MSE\
       --loss_weights 10.0 \
       --optimizer adam \
       --gan_path ../run/result_GAN_201214_testrun_alljapan/model_150.pt\

