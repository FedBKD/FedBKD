# Modified from: https://github.com/lgcollins/FedRep.git

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import logging
import time
import json

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from utils.train_utils import get_data, get_model, read_data
from models.Update import LocalUpdate, LocalUpdateDFGAN
from models.test import test_img_local_all
from log_utils.logger import loss_logger, cfs_mtrx_logger, parameter_logger, data_logger, para_record_dir,args
import os
from models.Nets import CNNCifarMulti, CNNCifar100Multi, MLPMulti, RNNSentMulti
from GAN_train import DF_gan

save_dir = "save_" + args.alg + '_' + args.dataset + '_' + str(args.num_users) \
               + '_' + str(args.shard_per_user) + "_" + str(args.attention)
para_dir = os.path.join(para_record_dir, args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
    args.shard_per_user) + "_" + str(args.attention) + "_" + str(args.seed))
log_file = './{}/metric.log'.format(save_dir)

if not os.path.exists('./{}'.format(save_dir)):
    os.mkdir('./{}'.format(save_dir))
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("")

if not os.path.exists(para_dir):
    os.mkdir(para_dir)

logging.basicConfig(filename=log_file, level=logging.DEBUG)

np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # parse args
    cuda0 = torch.device('cuda:' + str(args.gpu))
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if 'cifar' in args.dataset:
        shard_per_user = 's' + str(args.shard_per_user)
        ana_avg_model_save_path = './%s/%s/'%(args.dataset, shard_per_user)
    else:
        ana_avg_model_save_path = './%s/' % (args.dataset)

    print(args)
    loss_logger.info("Start experiment with args: \n {}".format(str(args)))
    cfs_mtrx_logger.info("Start experiment with args: \n {}".format(str(args)))
    parameter_logger.info("Start experiment with args: \n {}".format(str(args)))
    data_logger.info("Start experiment with args: \n {}".format(str(args)))

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test, user_label_dict = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = './' + args.dataset + '/data/train'
            test_path = './' + args.dataset + '/data/test'
        else:
            train_path = './' + args.dataset + '/data/train'
            test_path = './' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        user_label_dict = {}
        for ind, data in enumerate(dataset_train.values()):
            user_label_dict[ind] = list(set(data['y']))
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    print(args.alg)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    if args.alg == 'fedbkd' or args.alg == 'fedrep' or args.alg:
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [4, 3, 0, 1]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 6, 7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    if args.alg == 'fedbkd' or args.alg == 'fedrep' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    w_locals = {}
    w_locals_with_global_para = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
        w_locals_with_global_para[user] = copy.deepcopy(w_local_dict)

    # training
    indd = None
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()

    # print(idxs_users)
    client_sample_history = dict()
    acc_list = []
    acc_list_ = []
    next_sample_list = []
    next_sample_w_list = []
    for iter in range(args.epochs + 1):
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs - 1:
            m = args.num_users

        data_logger.info("epoch: \n {}".format(str(iter)))

        epoch_start = time.time()
        w_glob = {}
        loss_locals = []

        parameters = {}

        if iter == 0:
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        else:
            idxs_users = next_sample_list
        data_logger.info("samples: \n {}".format(",".join([str(ele) for ele in idxs_users.tolist()])))
        client_sample_history[iter] = idxs_users.tolist()
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        index_dict = {}

        start_time = time.time()
        for ind, idx in enumerate(idxs_users):
            index_dict[str(ind)] = idx
            start_in = time.time()
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if args.epochs == iter:
                    # finetune
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]],
                                        idxs=dict_users_train, indd=indd)
                else:
                    # train
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]],
                                        idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                if iter == 0:
                    w_local[k] = w_locals[idx][k]
                else:
                    if k in w_glob_keys:
                        w_local[k] = w_global[k]
                    else:
                        w_local[k] = w_locals[idx][k]

            net_local.load_state_dict(w_local)

            last = iter == args.epochs
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_local, loss, indd = local.train(net=net_local.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr, last=last)


            loss_locals.append(copy.deepcopy(loss))
            time_train_end = time.time()
            total_len += lens[idx]

            index = 0
            if len(w_glob) == 0:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        if key not in parameters.keys():
                            parameters[key] = []
                        parameters[key].append(w_local[key])
                        index += 1
                    w_locals[idx][key] = w_local[key]
                    w_locals_with_global_para[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        if key not in parameters.keys():
                            parameters[key] = []
                        parameters[key].append(w_local[key])
                        index += 1
                    w_locals[idx][key] = w_local[key]
                    w_locals_with_global_para[idx][key] = w_local[key]
            times_in.append(time.time() - start_in)
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        for ind, idx in enumerate(idxs_users):
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_locals[idx])
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = (w_glob[key] * lens[idx]).to(cuda0)
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += (w_locals[idx][key] * lens[idx]).to(cuda0)
                    else:
                        w_glob[key] += (w_locals[idx][key] * lens[idx]).to(cuda0)
        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)
        net_glob.load_state_dict(w_glob)

        w_global = copy.deepcopy(w_glob)

        if args.dataset == 'cifar100':
            net_per = CNNCifar100Multi(args=args).to(args.device)
        elif args.dataset == 'cifar10':
            net_per = CNNCifarMulti(args=args).to(args.device)
        elif args.dataset == 'sent140':
            net_per = RNNSentMulti(args=args).to(args.device)
        else:
            net_per = MLPMulti(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
        if iter > 0:
            y_input, eps, label = df_gan.y_input, df_gan.eps, df_gan.label
        else:
            y_input, eps, label  = None, None, None

        pre_model = None

        df_gan = DF_gan(args=args, seed=args.seed, w_locals=w_locals, selected_users=idxs_users,
                        dataset_test=dataset_test, dict_users_test=dict_users_test, user_label_dict=user_label_dict, y_input=y_input, eps=eps, label=label)
        df_gan.train_generator(epoches=args.gan_eps, pre_model=pre_model)

        global_dataset, user_train_index = df_gan.generate_data()

        net_per.load_state_dict(w_glob)

        tmp_global_model = copy.deepcopy(net_per)
        for ind,idx in enumerate(idxs_users):
            gen_train_index = user_train_index[idx]
            user_global_dataset = {}
            user_global_dataset['x'] = global_dataset['x'][gen_train_index[0]].unsqueeze(dim=0)
            user_global_dataset['y'] = global_dataset['y'][gen_train_index[0]].unsqueeze(dim=0)
            for i in gen_train_index[1:]:
                user_global_dataset['x'] = torch.cat((user_global_dataset['x'], global_dataset['x'][i].unsqueeze(dim=0)), dim=0)
                user_global_dataset['y'] = torch.cat(
                    (user_global_dataset['y'], global_dataset['y'][i].unsqueeze(dim=0)), dim=0)

            # local-->global
            global_model = LocalUpdateDFGAN(args=args, dataset=user_global_dataset)

            w_idx_local, loss = global_model.train(net_per=net_per.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr, w_locals=w_locals, idx=idx)

            # global-->local
            tmp_global_model.load_state_dict(w_global)
            w_avg_global, loss_avg = global_model.train(net_per=tmp_global_model.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr, w_locals=w_locals,
                               idx=idx, global_kd_local=True)

            w_global = w_avg_global
            for k in net_glob.state_dict().keys():
                w_locals[idx][k] = w_idx_local[k]

        next_sample_list = np.random.choice(range(args.num_users), m, replace=False)

        #
        net_glob.load_state_dict(w_global)
        if args.save_avg_model == 1:
            model_save_name = ana_avg_model_save_path + 'after_iter_%d_model_global_gen_lr1.pth'%(iter)
            torch.save(net_glob.state_dict(), model_save_name)

        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            if args.dataset == 'sent140':
                args.local_bs = 4
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                     w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                     dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                     return_all=False, iter=iter, idx_users=idxs_users)
            loss_logger.info("averaged local acc of round {}: \n{}".format(iter, json.dumps(acc_test)))
            loss_logger.info("averaged local loss of round {}: \n{}".format(iter, json.dumps(loss_test)))

            accs.append(acc_test)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))

                logging.info('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}\n'.format(
                    loss_avg, loss_test, acc_test))

            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10 += acc_test / 10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                         w_locals=None, indd=indd, dataset_train=dataset_train,
                                                         dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    print(
                        'Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            iter, loss_avg, loss_test, acc_test))
                else:
                    print(
                        'Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
                    loss_logger.info("averaged local acc of round {}: \n{}".format(str(iter) + " after para aggregate",
                                                                                   json.dumps(acc_test)))
                    loss_logger.info("averaged local loss of round {}: \n{}".format(str(iter) + " after para aggregate",
                                                                                json.dumps(loss_test)))

                loss_logger.info("averaged local acc of round {}: \n{}".format(iter, json.dumps(acc_test)))
                loss_logger.info("averaged local loss of round {}: \n{}".format(iter, json.dumps(loss_test)))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10_glob += acc_test / 10

        if iter % args.save_every == args.save_every - 1:
            model_save_path = './save/accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter' + str(iter) + args.function + '.pt'
            torch.save(net_glob.state_dict(), model_save_path)
        print("each epoch  cost time:%s" % str(time.time() - epoch_start))

    data_logger.info("client sample history: \n{}".format(json.dumps(client_sample_history)))
    data_logger.info("client update before performance:")
    for uid in range(args.num_users):
        uid_accs = {}
        sample_list = []
        for i in range(len(acc_list)):
            uid_accs[i] = acc_list[i][uid]
            if uid in client_sample_history[i]:
                sample_list.append(str(i))
        data_logger.info("client %s sampled history:" % str(uid))
        data_logger.info(",".join(sample_list))
        data_logger.info("client %s update before performance:" % str(uid))
        data_logger.info(json.dumps(uid_accs))
        data_logger.info("------------------------------\n")

    print('Average accuracy final 10 rounds: {}'.format(accs10))
    logging.info('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
        args.shard_per_user) + args.function + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
    logging.info("loss train")
    logging.info(",".join([str(e) for e in loss_train]) + "/n")
    logging.info("accs")
    logging.info(",".join(list(accs)) + "/n")
