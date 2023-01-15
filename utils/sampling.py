#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid2(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # 从随机取改为按顺序划分
    # divide and assign
    '''
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    '''
    shards_in_one = num_shards // num_users
    for i in range(num_users):
        dict_users[i] = idxs[i * shards_in_one * num_imgs:(i + 1) * shards_in_one * num_imgs].copy()

    return dict_users




def mnist_noniid(dataset, num_users, iidpart = 0.1):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    #取10%的数据为iid
    iid_idx = np.array([],dtype=idxs.dtype)
    noniid_idx = np.array([],dtype=idxs.dtype)

    iid_num = int(num_shards * num_imgs * iidpart)
    noniid_num = num_shards * num_imgs - iid_num
    for i in range(10):
        iid_idx = np.hstack((iid_idx, idxs[num_shards * num_imgs//10*i : num_shards * num_imgs//10*i + iid_num//10]))
        noniid_idx = np.hstack(
            (noniid_idx, idxs[num_shards * num_imgs // 10 * i + iid_num//10 : num_shards * num_imgs // 10 * (i+1)]))
    np.random.shuffle(iid_idx)

    shards_in_noniid_one = noniid_num//num_users
    shards_in_iid_one = iid_num // num_users
    for i in range(num_users):
        dict_users[i] = np.hstack((noniid_idx[i*shards_in_noniid_one: ( i + 1)*shards_in_noniid_one].copy(),
                                   iid_idx[i * shards_in_iid_one : (i + 1) * shards_in_iid_one].copy()))

    return dict_users




def mnist_noniid_multi(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_shards, num_imgs = num_users*2, 30000//num_users
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users




def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users, iidpart = 0.1):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    #取10%的数据为iid
    iid_idx = np.array([],dtype=idxs.dtype)
    noniid_idx = np.array([],dtype=idxs.dtype)

    iid_num = int(num_shards * num_imgs * iidpart)
    noniid_num = num_shards * num_imgs - iid_num
    for i in range(10):
        iid_idx = np.hstack((iid_idx, idxs[num_shards * num_imgs//10*i : num_shards * num_imgs//10*i + iid_num//10]))
        noniid_idx = np.hstack(
            (noniid_idx, idxs[num_shards * num_imgs // 10 * i + iid_num//10 : num_shards * num_imgs // 10 * (i+1)]))
    np.random.shuffle(iid_idx)

    shards_in_noniid_one = noniid_num//num_users
    shards_in_iid_one = iid_num // num_users
    for i in range(num_users):
        dict_users[i] = np.hstack((noniid_idx[i*shards_in_noniid_one: ( i + 1)*shards_in_noniid_one].copy(),
                                   iid_idx[i * shards_in_iid_one : (i + 1) * shards_in_iid_one].copy()))

    return dict_users

def cifar_noniid2(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 从随机取改为按顺序划分
    '''    
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)    
    '''
    shards_in_one = num_shards // num_users
    for i in range(num_users):
        dict_users[i] = idxs[i * shards_in_one * num_imgs:(i + 1) * shards_in_one * num_imgs]  # .copy()
        # random.shuffle(dict_users[i])

    return dict_users

def cifar_noniid_Dirichlet(data, num_users, seed, alpha):

    dict_users = {i: np.array([]) for i in range(num_users)}
    # labels = dataset.train_labels.numpy()
    #labels = np.array(data.targets)

    n_nets = num_users
    K = 10
    labelList = np.array(data.targets)
    min_size = 0
    N = len(labelList)
    np.random.seed(2020)

    net_dataidx_map = {}
    while min_size < K:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(labelList == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print('Data statistics: %s' % str(net_cls_counts))

    local_sizes = []
    for i in range(n_nets):
        local_sizes.append(len(net_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes / np.sum(local_sizes)
    print(weights)

    for i in range(num_users):
        dict_users[i] = idx_batch[i]
    return dict_users, weights




def cifar_noniid_multi(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users*2, 25000//num_users
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_shards//num_users, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)

        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    return dict_users


if __name__ == '__main__':
    #dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
    #                               transform=transforms.Compose([
    #                                   transforms.ToTensor(),
    #                                   transforms.Normalize((0.1307,), (0.3081,))
    #                               ]))
    #num = 100
    #d = mnist_noniid(dataset_train, num)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset_train = datasets.CIFAR10('../../data/', train=True, download=False, transform=transform_train)
    d, k = cifar_noniid_Dirichlet(dataset_train, 5, 1, 0.5)