FedBKD:Distilled Federated Learning to Embrace Gerneralization and Personalization on Non-IID Data

## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.2.0. To install the other dependencies: `pip3 install -r requirements.txt`.


## Usage

FedBKD is run using a command of the following form:

`python main_fedbkd.py --alg fedbkd --dataset [dataset] --num_users [num_users] --model [model] --model [model] --shard_per_user [shard_per_user] --frac [frac] --local_bs [local_bs] --lr [lr] --epochs [epochs] --local_ep [local_ep] --local_rep_ep [local_rep_ep] --gpu [gpu] --seed [seed] --server_glo_eps [server_glo_eps]`

Explanation of parameters:

- `alg` : algorithm to run, may be `fedbkd`, `fedrep`, `fedavg`, `prox` (FedProx), `fedper` (FedPer), or `lg` (LG-FedAvg)
- `dataset` : dataset, may be `cifar10`, `cifar100`, `femnist`, `mnist`, `sent140`
- `num_users` : number of users
- `model` : for the CIFAR datasets, we use `cnn`, for the MNIST datasets, we use `mlp`, and for `sent140`, we use `res`
- `num_classes` : total number of classes
- `shard_per_user` : number of classes per user (specific to CIFAR datasets and MNIST)
- `frac` : fraction of participating users in each round (for all experiments we use 0.1)
- `local_bs` : batch size used locally by each user
- `lr` : learning rate
- `epochs` : total number of communication rounds
- `local_ep` : total number of local epochs
- `local_rep_ep` : number of local epochs to execute for the representation (specific to FedBKD)
- `gpu` : GPU ID
- `seed`: random seed
- `server_glo_eps`: total number of global task rounds

A full list of configuration parameters and their descriptions are given in `utils/options.py`.
For examples of running FedBKD as well as the FL baselines we compare against, please see the executable files in `scripts/`, which recover the results from the paper.


# Acknowledgements

Much of the code in this repository was adapted from code in the repository https://github.com/lgcollins/FedRep.git
