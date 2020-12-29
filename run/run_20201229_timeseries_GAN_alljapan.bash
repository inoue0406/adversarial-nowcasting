#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

case="result_20201229_timeseries_GAN_alljapan"

# running script for Rainfall Prediction with time series prediction mode

python ../src/main_TS_seq2seq_jma.py --model_name seq2seq\
       --data_path result_20201227_gan_convert_to_latent/\
       --valid_data_path result_20201227_gan_convert_to_latent/\
       --train_path ../data/gan_latent_alljapan_2015-2016_small.csv\
       --valid_path ../data/gan_latent_alljapan_2015-2016_small.csv\
       --test --eval_threshold 0.5 10 20 \
       --test_path ../data/gan_latent_alljapan_2015-2016_small.csv\
       --result_path $case --tdim_use 12 --learning_rate 0.002 --lr_decay 0.9\
       --batch_size 20 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE\
       --optimizer adam\

