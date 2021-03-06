# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate, LocalUpdateAPFL
from models.test import test_img_local_all

import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = '/home/FedRep/data/' + args.dataset + '/data/train'
            test_path = '/home/FedRep/data/' + args.dataset + '/data/test'
        else:
            train_path = '/home/FedRep/data/' + args.dataset + '/data/train'
            test_path = '/home/FedRep/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(len(lens))
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    print(total_num_layers)

    # generate list of local models0106 for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] =net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    test_freq = args.test_freq
    indd = None
    accs = []
    loss_train = []
    acc_avg10 = 0
    alpha = args.alpha_apfl
    start = time.time()
    accs10 = 0
    for iter in range(args.epochs+1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for ind, idx in enumerate(idxs_users):
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                local  = LocalUpdateAPFL(args=args, dataset=dataset_train[list(dataset_train.keys())[idx]], idxs=dict_users_train, indd=indd)
            else:
                local = LocalUpdateAPFL(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
            net_local = copy.deepcopy(net_glob)
            w_local = copy.deepcopy(w_locals[idx])
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                # w_global, w_local, loss, indd = local.train(net=net_local.to(args.device),w_local=w_local, ind=idx, idx=clients[idx], w_glob_keys=w_glob_keys, lr=args.lr)
                w_global, w_local, loss, indd = local.train(net=net_local.to(args.device), w_local=w_local, ind=idx, lr=args.lr)
            else:
                w_global, w_local, loss, indd = local.train(net=net_local.to(args.device),w_local=w_local, idx=idx, lr=args.lr)
            loss_locals.append(copy.deepcopy(loss))

            if len(w_glob) == 0:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_global[key]/m
                    w_locals[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_global[key]/m
                    w_locals[idx][key] = w_local[key]
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        # get weighted average for global weights
        net_glob.load_state_dict(w_glob)
        
        w_locals_test = copy.deepcopy(w_locals)
        for user in range(args.num_users):
            for key in w_locals[user].keys():
                w_locals_test[user][key] = alpha*w_locals[user][key] + (1-alpha)*w_glob[key] 

        if iter % args.test_freq==args.test_freq-1 or iter>=args.epochs-10:
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                     w_locals=w_locals_test, indd=indd, dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
            accs.append(acc_test)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))
            if iter >= args.epochs-10 and iter != args.epochs:
                accs10 += acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_apfl_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    end = time.time()
    print(end-start)
    print(accs)
    base_dir = './save/accs_apfl_' +  args.dataset + str(args.num_users) +'_'+ str(args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
