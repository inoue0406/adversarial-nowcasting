#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

case="result_20201004_traj_clstm_size200_alljapan_mse_lr0002"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name clstm\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --aug_rotate 0 --aug_resize 0\
       --data_path ../data/data_alljapan/ --image_size 200\
       --valid_data_path ../data/data_alljapan/ \
       --train_path ../data/train_alljapan_2yrs_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv\
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv\
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0002 --lr_decay 0.9\
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function MSE\
       --hidden_channels 5 --kernel_size 3 --optimizer adam \
# post plot
python ../src_post/plot_comp_prediction_trajgru.py $case
python ../src_post/gif_animation.py $case

case="result_20201004_traj_clstm_size200_alljapan_wmse_lr0002"

# running script for Rainfall Prediction with ConvLSTM
python ../src/main_trajGRU_jma.py --model_name clstm\
       --dataset radarJMA --model_mode run --data_scaling linear\
       --aug_rotate 0 --aug_resize 0\
       --data_path ../data/data_alljapan/ --image_size 200\
       --valid_data_path ../data/data_alljapan/ \
       --train_path ../data/train_alljapan_2yrs_JMARadar.csv \
       --valid_path ../data/valid_simple_JMARadar.csv\
       --test --eval_threshold 0.5 10 20 --test_path ../data/valid_simple_JMARadar.csv\
       --result_path $case --tdim_use 12 --tdim_loss 12 --learning_rate 0.0002 --lr_decay 0.9\
       --batch_size 10 --n_epochs 20 --n_threads 4 --checkpoint 10 \
       --loss_function WeightedMSE\
       --hidden_channels 5 --kernel_size 3 --optimizer adam \
# post plot
python ../src_post/plot_comp_prediction_trajgru.py $case
python ../src_post/gif_animation.py $case



