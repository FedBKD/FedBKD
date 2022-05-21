FedBKD:Distilled Federated Learning to Embrace Gerneralization and Personalization on Non-IID Data

## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.2.0. To install the other dependencies: `pip3 install -r requirements.txt`.


## Usage

FedBKD is run using a command of the following form:

`python main_fedbkd.py --alg fedbkd --dataset [dataset] --num_users [num_users] --model [model] --model [model] --shard_per_user [shard_per_user] --frac [frac] --local_bs [local_bs] --lr [lr] --epochs [epochs] --local_ep [local_ep] --local_rep_ep [local_rep_ep] --gpu [gpu] --seed [seed] --server_glo_eps [server_glo_eps] --server_rep_eps [server_rep_eps] --gen_eps [gen_eps] --gen_train_nums [gen_train_nums] --tem [tem]`


A full list of configuration parameters and their descriptions are given in `utils/options.py`.
For examples of running FedBKD as well as the FL baselines we compare against, please see the executable files in `scripts/`, which recover the results from the paper.


# Acknowledgements

Much of the code in this repository was adapted from code in the repository https://github.com/lgcollins/FedRep.git and https://github.com/zhuangdizhu/FedGen.git
