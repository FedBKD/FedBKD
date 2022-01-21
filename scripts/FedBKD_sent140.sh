for RUN in 1 2 3 4 5; do
  python main_fedbkd.py --dataset sent140 --model res --epochs 100 --alg fedbkd --lr 0.01 \
  --num_users 183 --gpu 0 --test_freq 50 --local_ep 15 --frac 0.1 --local_rep_ep 5 --local_bs 4 --server_glo_eps 5

done