#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torchfusion.datasets as fdatasets
import torch

from utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_multi, cifar_iid, cifar_noniid, cifar_noniid2, cifar_noniid_multi
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
import models.Fed as Fed
from models.test import test_img
from models.vgg import VGG
from models.lenet import LeNet
import models.resnet as resnet
import models.resnet_gn as resnet_gn
import models.densenet as densenet
import utils.util as util
import random
import utils.fmnist as fmnist

from tensorboardX import SummaryWriter
import utils.summary as summary

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def change_lr(lr, current_epoch):
    if current_epoch >= 60:
        return 0.01
    elif current_epoch >= 100:
        return 0.001
    else:
        return lr

def model_minus(w1, w2):
    minus = {}
    for n, p in w1.items():
        minus[n] = (p - w2[n])
    return minus

def model_distance(w1, w2):
    dis = torch.tensor(0.0)
    for n,p in w1.items():
        if ('running' not in n) and ('track' not in n):
            dis1 = (w1[n] - w2[n]) ** 2
            dis = dis + dis1.sum()
    return dis.sqrt().cpu().numpy()

def vector_cos(w1, w2):
    numerator = torch.tensor(0.0)
    denominator1 = torch.tensor(0.0)
    denominator2 = torch.tensor(0.0)
    for n, p in w1.items():
        numerator = numerator + (p*w2[n]).sum()
        denominator1 = denominator1 + (p*p).sum().sqrt()
        denominator2 = denominator2 + (w2[n]*w2[n]).sum().sqrt()
    return numerator/(denominator1*denominator2)

if __name__ == '__main__':
    setup_seed(111)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    logger = SummaryWriter(args.summary_path)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fmnist':
        #_train = fmnist.load_mnist('../data/fmnist/', kind='train')
        #dataset_test = fmnist.load_mnist('../data/fmnist/', kind='t10k')
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = fdatasets.FashionMNIST('../data/fmnist/',train=True, transform=trans_fmnist, download=False)
        dataset_test = fdatasets.FashionMNIST('../data/fmnist/', train=False, transform=trans_fmnist, download=False)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:

            if args.num_users > 10:
                dict_users = mnist_noniid_multi(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users, iidpart=0.3)

    elif args.dataset == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/', train=True, download=False, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/', train=False, download=False, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            if args.num_users > 10:
                dict_users = cifar_noniid_multi(dataset_train, args.num_users)
            else:
                dict_users = cifar_noniid2(dataset_train, args.num_users)
                # dict_users = cifar_noniid(dataset_train, args.num_users, 0.1)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        if args.dataset == 'cifar':
            if args.model == 'cnn':
                net_glob = CNNCifar(args=args).to(args.device)
            elif args.model == 'vgg16':
                net_glob = VGG('VGG16', momentum=0.1, m=args.num_users).to(args.device)
            elif args.model == 'resnet18':
                #net_glob = resnet.ResNet18().to(args.device)
                net_glob = resnet_gn.resnet18(group_norm=16).to(args.device)
            elif args.model == 'densenet':
                net_glob = densenet.densenet_cifar().to(args.device)
            elif args.model == 'lenet':
                net_glob = LeNet(momentum=0.1, m=args.num_users, selfbn=args.selfbn, device=args.device).to(args.device)
        elif args.dataset == 'mnist' or args.dataset == 'fmnist':
            if args.model == 'cnn':
                net_glob = CNNMnist(args=args).to(args.device)

    #net_glob = convert_bn_model_to_gn(net_glob)
    #batchlayer_klist = util.batchnorm_layer_k(net_glob)
    print(net_glob)
    #summary.summary(net_glob, input_size=(3, 32, 32), device=args.device)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    w_1order_glob = {}
    w_1order_glob_old = {}
    w_locals = []
    fisher_locals = []


    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
        fisher_locals = [w_glob for i in range(args.num_users)]

    clients = []
    for idx in range(args.num_users):
        clients.append(LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], worker_id=idx, logger=logger))

    change_mask = {}
    server_momentum = {}


    current_epoch = 0
    global_round = 0
    step_in_round = args.step_in_round_init
    lr = args.lr
    while current_epoch < args.epochs:
    #while global_round < 250:
    #for iter in range(args.epochs):
        #net_glob.apply(print_bn)

        #lr = change_lr(lr, current_epoch)

        net_glob.train()
        loss_locals = []
        acc_locals = []
        dis_locals = []
        offset_locals = []
        if not args.all_clients:
            w_locals = []
            fisher_locals = []
        if not args.all_clients:
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        else:
            idxs_users = range(args.num_users)




        for idx in idxs_users:
            local = clients[idx]
            w, fisher, loss, acc, dis, offset = local.train(net=copy.deepcopy(net_glob).to(args.device), step_in_round=step_in_round,
                                       global_round=global_round, lr=lr, args=args, change_mask=change_mask, w_1order_glob=w_1order_glob)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
                fisher_locals[idx] = copy.deepcopy(fisher)
            else:
                w_locals.append(copy.deepcopy(w))
                fisher_locals.append(copy.deepcopy(fisher))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(acc)
            dis_locals.append(dis)
            offset_locals.append(offset)
            # suppose all the workers have the same num of batches
            current_epoch = local.get_epoch()
            past_batch_in_epoch = local.get_past_batch()
            step_in_epoch = local.get_step_per_epoch()
            global_step = local.get_step()




        global_round += 1
        step_in_round = args.step_in_round

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        acc_avg = sum(acc_locals) / len(acc_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(global_round, loss_avg))
        #loss_train.append(loss_avg)
        record_step = global_round * step_in_round
        if args.all_clients:
            record_step = global_step


        # 计算模型与模型0的距离
        for j in range(1, len(w_locals)):
            dis = model_distance(w_locals[0], w_locals[j])
            cos = (dis_locals[0] ** 2 + dis_locals[j] ** 2 - dis ** 2) / (2 * dis_locals[0] * dis_locals[j])
            print('model 0 <--dis--> model {}: {:.6f}, cos={:.6f}'.format(j, dis, cos))
            if j == 1:
                logger.add_scalar('model0<cos>model1', cos, record_step)
                logger.add_scalar('model0<dis>model1', dis, record_step)


        logger.add_scalar('loss/train', loss_locals[-1], record_step)
        logger.add_scalar('acc/train', 100 * acc_locals[-1], record_step)

        # copy weight to net_glob
        w_glob_old = copy.deepcopy(w_glob)
        w_glob = Fed.FedAvg(w_locals)


        if len(w_1order_glob) > 0:
            w_1order_glob_old = w_1order_glob
        w_1order_glob = model_minus(w_glob, w_glob_old)

        if len(w_1order_glob_old) > 0:
            print('last_step<----cos--->this step: {:.6f}'.format(vector_cos(w_1order_glob_old, w_1order_glob)))

        if args.method == 8:
            if global_round < 5:
                rate = 1
            else:
                rate = args.imp1 - 1 / (global_round - 4)
            for k, v in w_glob_old.items():
                w_glob[k] = v + rate * w_1order_glob[k]
        else:
            w_glob = Fed.server_opt(w_glob_old, w_1order_glob, server_momentum, momentum=args.server_mom)

        global_move = model_distance(w_glob_old, w_glob)
        print('last_model <--dis--> model: {:.6f}'.format(global_move))
        logger.add_scalar('global_move', global_move, record_step)
        net_glob.load_state_dict(w_glob)

        if True:#(args.step_in_round * global_round) % 100 == 0:
            net_glob.eval()

            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("epoch {} [{:.0f}] Fed {} Testing accuracy: {:.2f}".format(current_epoch, past_batch_in_epoch*1.0 / step_in_epoch, 0, acc_test))


            logger.add_scalar('loss/test'+str(i), loss_test, record_step)
            logger.add_scalar('acc/test'+str(i), acc_test, record_step)
