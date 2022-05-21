#!/bin/bash

for RUN in 1 2 3 4 5; do
  python main_fedbkd.py --dataset cifar10 --model cnn --num_classes 10 --epochs 100 --alg fedbkd --lr 0.01 \
  --num_users 100 --gpu 0 --shard_per_user 2 --test_freq 50 --local_ep 11 --frac 0.1 --local_rep_ep 1 --local_bs 10 \
  --server_glo_eps 4 --server_rep_eps 1 --gen_eps 6 --gen_train_nums 1000 --tem 10

done
