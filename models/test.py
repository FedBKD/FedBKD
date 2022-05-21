# Modified from: https://github.com/lgcollins/FedRep.git

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from models.language_utils import repackage_hidden, process_x, process_y
import json

from log_utils.logger import loss_logger, cfs_mtrx_logger


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label

def test_img_local(net_g, dataset, args, idx=None, indd=None, user_idx=-1, idxs=None, iter=None):
    net_g.eval()
    test_loss = 0
    correct = 0
    confusion_martix = np.zeros([args.num_classes, args.num_classes])
    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf = True
        datatest_new = []
        usr = idx
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append(
                (torch.reshape(torch.tensor(dataset[idx]['x'][j]), (1, 28, 28)), torch.tensor(dataset[idx]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf = True
        datatest_new = []
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j], dataset[idx]['y'][j]))
    else:
        leaf = False

    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new, np.ones(len(datatest_new))), batch_size=args.local_bs,
                                 shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=False)
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        if 'sent140' in args.dataset:
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, hidden_train = net_g(data, hidden_train)

            loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
            _, pred_label = torch.max(output.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            # count += args.local_bs
            count += min(args.local_bs, input_data.shape[0])
            # print(args.local_bs)
            # print(count)
            test_loss += loss.item() * min(args.local_bs, input_data.shape[0])

        else:
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            for tgt, prd in zip(target.tolist(), y_pred.squeeze(1).tolist()):
                confusion_martix[tgt][prd] += 1

            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return accuracy, test_loss, confusion_martix


def test_img_local_all(net, args, dataset_test, dict_users_test,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False, iter=None, idx_users=None):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    acc_test_local_record = np.zeros(num_idxxs)
    loss_test_local_record = np.zeros(num_idxxs)
    confusion_martix_record = {i: None for i in range(num_idxxs)}
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if idx_users is not None:
            w_local = net_local.state_dict()
            if idx in idx_users:
                for k in w_locals[idx].keys():
                    w_local[k] = w_locals[idx][k]
            else:
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
        else:
            if w_locals is not None:
                w_local = net_local.state_dict()
                for k in w_locals[idx].keys():
                    w_local[k] = w_locals[idx][k]
        net_local.load_state_dict(w_local)
        net_local.eval()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            # print("---------user----------")
            # print(dict_users_test[idx])
            a, b, confusion_martix =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
            tot += len(dataset_test[dict_users_test[idx]]['x'])
            # print(len(dataset_test[dict_users_test[idx]]['x']))
        else:
            a, b, confusion_martix = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx], iter=iter)
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local_record[idx] = a
            loss_test_local_record[idx] = b
            confusion_martix_record[idx] = confusion_martix.tolist()
            acc_test_local[idx] = a*len(dict_users_test[idx])
            loss_test_local[idx] = b*len(dict_users_test[idx])
        del net_local

    if iter is not None:
        loss_logger.info("local acc of round {}: \n{}".format(iter,
                                                              {ii : acc for ii, acc in enumerate(acc_test_local_record.tolist())}
                                                              ))
        loss_logger.info("local acc of round {}: \n{}".format(iter, json.dumps(acc_test_local_record.tolist())))
        loss_logger.info("local loss of round {}: \n{}".format(iter, json.dumps(loss_test_local_record.tolist())))
        cfs_mtrx_logger.info("local confusion martix of round {}: \n{}".format(iter, json.dumps(confusion_martix_record)))

    if return_all:
        return acc_test_local, loss_test_local
    return sum(acc_test_local)/tot, sum(loss_test_local)/tot
