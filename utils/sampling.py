#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import random
import numpy as np
import torch
import copy

def noniid_artificial(dataset, num_users, shard_per_user, num_classes, seed, rand_set_all=[], testb=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 按user索引

    idxs_dict = {}  # 按label类索引, 每个key下放着该label所有数据的idx
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)  # 每一类应该被分成多少小块
    samples_per_user = int( count/num_users )  # 每一个client应该有多少数据
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x  # label下x为list，x长度为每一类应该被分成小块数量，x每个元素为包含data idx的list

    if len(rand_set_all) == 0:
        rand_set_all = []  # shard_per_user * num_users
        for i in range(0, num_classes, shard_per_user):
            rand_set_all += list(range(i, i + shard_per_user)) * shard_per_class
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
        # list,分num_users部分，决定每个user上的类分布，每一部分放着一个user上所有数据点的label

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]  # 每一个user上的labels
        rand_set = []
        for label in rand_set_label:
            np.random.seed(seed + label * 1000)
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))  # user拿走这个label下的一小块
        dict_users[i] = np.concatenate(rand_set)
        # dict，按user idx索引，每个vale放这个user idx下存储的所有data的idx

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all

def noniid_artificial_imbalance_v2(dataset, num_users, shard_per_user, num_classes, seed, rand_set_all=[], testb=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    labelinfo_list = []
    labelinfo_dic = {}
    label_list = [0,1,2,3,4,5,6,7,8,9]
    np.random.seed(10086)
    for i in range(num_users):
        client_labelnum = np.random.randint(2,6)
        client_labelinfo = np.random.choice(label_list, client_labelnum, replace=False)
        for item in client_labelinfo:
            if item not in labelinfo_dic.keys():
                labelinfo_dic[item] = []
            labelinfo_dic[item].append(i)
        labelinfo_list.append(client_labelinfo.tolist())

    idxs_dict = {}  # 按label类索引, 每个key下放着该label所有数据的idx
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)

    dict_users = {i: [] for i in range(num_users)}  # 按user索引
    for item in labelinfo_dic.keys():
        data_ids = idxs_dict[item]
        client_ids = labelinfo_dic[item]
        for i in data_ids:
            client_id = client_ids[np.random.randint(0,len(client_ids))]
            dict_users[client_id].append(i)

    for i in dict_users.keys():
        dict_users[i] = np.array(dict_users[i])
    #     print (dict_users[i])
    # dict_users = {i: np.array(dict_users[i]) for i in dict_users.keys}

    return dict_users, labelinfo_list


# # brianzhao add global data 20211222 predict
# def noniid_global(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb=False):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     np.random.seed(20)
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     idxs_dict = {}
#     idxs_global = []
#     count = 0
#     for i in range(len(dataset)):
#         label = torch.tensor(dataset.targets[i]).item()
#         if label < num_classes and label not in idxs_dict.keys():
#             idxs_dict[label] = []
#         if label < num_classes:
#             idxs_dict[label].append(i)
#             count += 1
#
#     sample_seed = np.random.uniform(0, 0.005, 100)
#     np.random.shuffle(sample_seed)
#     sample_ratio = np.random.choice(sample_seed, 10)
#     sample_num = [math.ceil(i) for i in len(dataset)*sample_ratio]
#
#     for ind, (k, v) in enumerate(idxs_dict.items()):
#         global_sample_id = np.random.choice(range(len(v)), sample_num[ind], replace=False)
#         global_sample_idx = [v[i] for i in global_sample_id]
#         idxs_dict[k] = list(set(v)-set(global_sample_idx))
#         idxs_global.extend(global_sample_idx)
#         count -= len(global_sample_idx)
#     np.random.shuffle(idxs_global)
#
#     shard_per_class = int(shard_per_user * num_users / num_classes)
#     samples_per_user = int( count/num_users )
#     # whether to sample more test samples per user
#     if (samples_per_user < 100):
#         double = True
#     else:
#         double = False
#
#     for label in idxs_dict.keys():
#         x = idxs_dict[label]
#         num_leftover = len(x) % shard_per_class
#         leftover = x[-num_leftover:] if num_leftover > 0 else []
#         x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
#         x = x.reshape((shard_per_class, -1))
#         x = list(x)
#
#         for i, idx in enumerate(leftover):
#             x[i] = np.concatenate([x[i], [idx]])
#         idxs_dict[label] = x
#
#     if len(rand_set_all) == 0:
#         rand_set_all = list(range(num_classes)) * shard_per_class
#         random.shuffle(rand_set_all)
#         rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
#
#     # divide and assign
#     for i in range(num_users):
#         if double:
#             rand_set_label = list(rand_set_all[i]) * 50
#         else:
#             rand_set_label = rand_set_all[i]
#         rand_set = []
#         for label in rand_set_label:
#             try:
#                 idx = np.random.choice(len(idxs_dict[label]), replace=False)
#             except:
#                 print('zzzz')
#             if (samples_per_user < 100 and testb):
#                 rand_set.append(idxs_dict[label][idx])
#             else:
#                 rand_set.append(idxs_dict[label].pop(idx))
#         dict_users[i] = np.concatenate(rand_set)
#
#     test = []
#     for key, value in dict_users.items():
#         x = np.unique(torch.tensor(dataset.targets)[value])
#         test.append(value)
#     test = np.concatenate(test)
#
#     return dict_users, rand_set_all, idxs_global

# # # brianzhao add global data 20211229  不要global
# def noniid_global(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb=False):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     idxs_dict = {}
#     count = 0
#     idxs_global = list(np.random.choice(range(len(dataset)), math.ceil(len(dataset)*0.01), replace=False))
#     for i in range(len(dataset)):
#         label = torch.tensor(dataset.targets[i]).item()
#         if label < num_classes and label not in idxs_dict.keys():
#             idxs_dict[label] = []
#         if label < num_classes:
#             idxs_dict[label].append(i)
#             count += 1
#
#     shard_per_class = int(shard_per_user * num_users / num_classes)
#     samples_per_user = int( count/num_users )
#     # whether to sample more test samples per user
#     if (samples_per_user < 100):
#         double = True
#     else:
#         double = False
#
#     for label in idxs_dict.keys():
#         x = idxs_dict[label]
#         num_leftover = len(x) % shard_per_class
#         leftover = x[-num_leftover:] if num_leftover > 0 else []
#         x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
#         x = x.reshape((shard_per_class, -1))
#         x = list(x)
#
#         for i, idx in enumerate(leftover):
#             x[i] = np.concatenate([x[i], [idx]])
#         idxs_dict[label] = x
#
#     if len(rand_set_all) == 0:
#         rand_set_all = list(range(num_classes)) * shard_per_class
#         random.shuffle(rand_set_all)
#         rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
#
#     # divide and assign
#     for i in range(num_users):
#         if double:
#             rand_set_label = list(rand_set_all[i]) * 50
#         else:
#             rand_set_label = rand_set_all[i]
#         rand_set = []
#         for label in rand_set_label:
#             try:
#                 idx = np.random.choice(len(idxs_dict[label]), replace=False)
#             except:
#                 print('zzzz')
#             if (samples_per_user < 100 and testb):
#                 rand_set.append(idxs_dict[label][idx])
#             else:
#                 rand_set.append(idxs_dict[label].pop(idx))
#         dict_users[i] = np.concatenate(rand_set)
#
#     test = []
#     for key, value in dict_users.items():
#         x = np.unique(torch.tensor(dataset.targets)[value])
#         test.append(value)
#     test = np.concatenate(test)
#
#     return dict_users, rand_set_all, idxs_global

# # # brianzhao add global data 20211222
# def noniid_global(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb=False):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     idxs_dict = {}
#     count = 0
#     idxs_global = list(np.random.choice(range(len(dataset)), math.ceil(len(dataset)*0.01), replace=False))
#     for i in range(len(dataset)):
#         if i not in idxs_global:
#             label = torch.tensor(dataset.targets[i]).item()
#             if label < num_classes and label not in idxs_dict.keys():
#                 idxs_dict[label] = []
#             if label < num_classes:
#                 idxs_dict[label].append(i)
#                 count += 1
#
#     shard_per_class = int(shard_per_user * num_users / num_classes)
#     samples_per_user = int( count/num_users )
#     # whether to sample more test samples per user
#     if (samples_per_user < 100):
#         double = True
#     else:
#         double = False
#
#     for label in idxs_dict.keys():
#         x = idxs_dict[label]
#         num_leftover = len(x) % shard_per_class
#         leftover = x[-num_leftover:] if num_leftover > 0 else []
#         x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
#         x = x.reshape((shard_per_class, -1))
#         x = list(x)
#
#         for i, idx in enumerate(leftover):
#             x[i] = np.concatenate([x[i], [idx]])
#         idxs_dict[label] = x
#
#     if len(rand_set_all) == 0:
#         rand_set_all = list(range(num_classes)) * shard_per_class
#         random.shuffle(rand_set_all)
#         rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
#
#     # divide and assign
#     for i in range(num_users):
#         if double:
#             rand_set_label = list(rand_set_all[i]) * 50
#         else:
#             rand_set_label = rand_set_all[i]
#         rand_set = []
#         for label in rand_set_label:
#             try:
#                 idx = np.random.choice(len(idxs_dict[label]), replace=False)
#             except:
#                 print('zzzz')
#             if (samples_per_user < 100 and testb):
#                 rand_set.append(idxs_dict[label][idx])
#             else:
#                 rand_set.append(idxs_dict[label].pop(idx))
#         dict_users[i] = np.concatenate(rand_set)
#
#     test = []
#     for key, value in dict_users.items():
#         x = np.unique(torch.tensor(dataset.targets)[value])
#         test.append(value)
#     test = np.concatenate(test)
#
#     return dict_users, rand_set_all, idxs_global


# # # brianzhao add global data 20211222   原始
# def noniid_global(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb=False):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#
#     idxs_dict = {}
#     idxs_global = []
#     count = 0
#     for i in range(len(dataset)):
#         label = torch.tensor(dataset.targets[i]).item()
#         if label < num_classes and label not in idxs_dict.keys():
#             idxs_dict[label] = []
#         if label < num_classes:
#             idxs_dict[label].append(i)
#             count += 1
#     for k, v in idxs_dict.items():
#         global_sample_id = np.random.choice(range(len(v)), len(v)//100, replace=False)
#         global_sample_idx = [v[i] for i in global_sample_id]
#         idxs_dict[k] = list(set(v)-set(global_sample_idx))
#         idxs_global.extend(global_sample_idx)
#         count -= len(global_sample_idx)
#     np.random.shuffle(idxs_global)
#
#     shard_per_class = int(shard_per_user * num_users / num_classes)
#     samples_per_user = int( count/num_users )
#     # whether to sample more test samples per user
#     if (samples_per_user < 100):
#         double = True
#     else:
#         double = False
#
#     for label in idxs_dict.keys():
#         x = idxs_dict[label]
#         num_leftover = len(x) % shard_per_class
#         leftover = x[-num_leftover:] if num_leftover > 0 else []
#         x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
#         x = x.reshape((shard_per_class, -1))
#         x = list(x)
#
#         for i, idx in enumerate(leftover):
#             x[i] = np.concatenate([x[i], [idx]])
#         idxs_dict[label] = x
#
#     if len(rand_set_all) == 0:
#         rand_set_all = list(range(num_classes)) * shard_per_class
#         random.shuffle(rand_set_all)
#         rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
#
#     # divide and assign
#     for i in range(num_users):
#         if double:
#             rand_set_label = list(rand_set_all[i]) * 50
#         else:
#             rand_set_label = rand_set_all[i]
#         rand_set = []
#         for label in rand_set_label:
#             try:
#                 idx = np.random.choice(len(idxs_dict[label]), replace=False)
#             except:
#                 print('zzzz')
#             if (samples_per_user < 100 and testb):
#                 rand_set.append(idxs_dict[label][idx])
#             else:
#                 rand_set.append(idxs_dict[label].pop(idx))
#         dict_users[i] = np.concatenate(rand_set)
#
#     test = []
#     for key, value in dict_users.items():
#         x = np.unique(torch.tensor(dataset.targets)[value])
#         test.append(value)
#     test = np.concatenate(test)
#
#     return dict_users, rand_set_all, idxs_global

# # brianzhao 取1000张数据  global任务随机采样
def noniid_global(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(10086)
    # np.random.seed(87684)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    idxs_global = []
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1
    for k, v in idxs_dict.items():
        # global_sample_id = np.random.choice(range(len(v)), len(v)//50, replace=False)
        global_sample_id = np.random.choice(range(len(v)), math.ceil(len(v) * 0.02), replace=False)
        global_sample_idx = [v[i] for i in global_sample_id]
        idxs_dict[k] = list(set(v)-set(global_sample_idx))
        idxs_global.extend(global_sample_idx)
        count -= len(global_sample_idx)
    np.random.shuffle(idxs_global)

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        np.random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            try:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
            except:
                print('zzzz')
            if (samples_per_user < 100 or testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all, idxs_global


def noniid(dataset, num_users, shard_per_user, num_classes, rand_set_all=[], testb=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    np.random.seed(10086)
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        np.random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            try:
                idx = np.random.choice(len(idxs_dict[label]), replace=False)
            except:
                print('zzzz')
            if (samples_per_user < 100 or testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    for key, value in dict_users.items():
        x = np.unique(torch.tensor(dataset.targets)[value])
        test.append(value)
    test = np.concatenate(test)

    return dict_users, rand_set_all

def noniid_artificial_imbalance(dataset, num_users, shard_per_user, num_classes, seed, rand_set_all=[], testb=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_distrib = num_classes // shard_per_user
    proportion_per_distrib = []
    for i in range(num_distrib):
        p = 2 ** (i // 2)
        proportion_per_distrib.append(p)
    sum_p = sum(proportion_per_distrib)
    proportion_per_distrib = [tmp / sum_p for tmp in proportion_per_distrib]
    enlarge = int((proportion_per_distrib)[-1] * len(proportion_per_distrib))


    num_users_enlarged = num_users * enlarge

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 按user索引

    idxs_dict = {}  # 按label类索引, 每个key下放着该label所有数据的idx
    count = 0
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users_enlarged / num_classes)  # 每一类应该被分成多少小块
    samples_per_user = int(count/num_users_enlarged)  # 每一个client应该有多少数据
    # whether to sample more test samples per user
    if (samples_per_user < 100):
        double = False
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x  # label下x为list，x长度为每一类应该被分成小块数量，x每个元素为包含data idx的list

    # print("=======proportion_per_distrib======")
    # print(proportion_per_distrib)
    # print(shard_per_class)

    if len(rand_set_all) == 0:
        rand_set_all = []  # shard_per_user * num_users
        for j, i in enumerate(range(0, num_classes, shard_per_user)):
            rand_set_all += list(range(i, i + shard_per_user)) * int(proportion_per_distrib[j] / proportion_per_distrib[-1] * shard_per_class)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))
        # list,分num_users部分，决定每个user上的类分布，每一部分放着一个user上所有数据点的label

    # print("====================idxs_dict check====================")
    # for k, v in idxs_dict.items():
    #     print(k, len(v))
    #
    # print("====================rand_set_all check====================")
    # print(rand_set_all)
    # raise NotImplementedError

    # divide and assign
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]  # 每一个user上的labels
        rand_set = []
        for label in rand_set_label:
            np.random.seed(seed + label * 1000)
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))  # user拿走这个label下的一小块
        dict_users[i] = np.concatenate(rand_set)
        # dict，按user idx索引，每个vale放这个user idx下存储的所有data的idx

    # print("====================dict_users check====================")
    # for k, v in dict_users.items():
    #     print(k, v.shape)
    # raise NotImplementedError

    return dict_users, rand_set_all