for RUN in 1 2 3 4 5; do
  python main_fedbkd.py --dataset cifar100 --model cnn --num_classes 100 --epochs 100 --alg fedbkd --lr 0.01 \
  --num_users 100 --gpu 0 --shard_per_user 20 --test_freq 50 --local_ep 15 --frac 0.1 --local_rep_ep 5 --local_bs 10 --server_glo_eps 5

done