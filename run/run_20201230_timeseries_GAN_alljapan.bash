#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

case="result_20201230_timeseries_GAN_alljapan"

# running script for Rainfall Prediction with time series prediction mode

python ../src/main_TS_seq2seq_jma.py --model_name seq2seq --no_train\
       --data_scaling linear\
       --data_path result_20201227_gan_convert_to_latent/\
       --valid_data_path result_20201227_gan_convert_to_latent/\
       --train_path ../data/gan_latent_alljapan_2015-2016_small.csv\
       --valid_path ../data/gan_latent_alljapan_2015-2016_small.csv\
       --test --eval_threshold 0.5 10 20 \
       --test_data_latent result_20201229_gan_convert_to_latent_kanto_valid/\
       --test_data_grid ../data/data_kanto/\
       --test_path ../data/gan_latent_kanto_2017_small.csv\
       --result_path $case --tdim_use 12 --learning_rate 0.002 --lr_decay 0.9\
       --batch_size 20 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE\
       --optimizer adam\
       --gan_path ../run/result_GAN_201214_testrun_alljapan/model_150.pt\

