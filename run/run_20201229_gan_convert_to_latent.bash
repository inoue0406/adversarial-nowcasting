#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1

case="result_20201229_gan_convert_to_latent_kanto_valid"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_GAN_convert_to_latent_jma.py \
       --dataset radarJMA --data_scaling linear\
       --data_path ../data/data_kanto/ --image_size 200\
       --train_path ../data/valid_simple_JMARadar_tmp_small.csv \
       --result_path $case --tdim_use 12 --learning_rate 0.1 --lr_decay 0.9\
       --batch_size 10 --n_epochs 20 --n_threads 4 \
       --loss_function MSE\
       --optimizer adam \
       --gan_path ../run/result_GAN_201214_testrun_alljapan/model_150.pt\

#       --train_path ../data/valid_simple_JMARadar.csv \
