#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=1000, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')


    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--all_clients', type=bool, default=True, help='aggregation over all clients')

    parser.add_argument('--selfbn', type=int, default=0, help='0: batchnorm')
    parser.add_argument('--method', type=int, default=9,
                        help='0: fedavg, 1:fedprox  7:fedcos 8:fedlen')
    parser.add_argument('--imp0', type=float, default=0.02, help='fedprox importance')
    parser.add_argument('--imp1', type=float, default=1, help='fedbatch importance')
    parser.add_argument('--step_in_round_init', type=int, default=400, help='step num in first round')
    parser.add_argument('--step_in_round', type=int, default=400, help='step num in each round')
    parser.add_argument('--server_mom', type=float, default=0.0, help='server momentum')
    #parser.add_argument('--summary_path', default='logs/mnist/cnn/noniid(4worker-60e-128bs-0.1lr-FedEWCAvg)',
    parser.add_argument('--summary_path', default='logs/cifar10/cnn/noniid(5worker-1frac-500e-128bs-400r-0.01lr-fedcos0.02)',
                        type=str,
                        help='model saved path')
    parser.add_argument('--uncertainty', type=int, default=0, help='0: none')
    args = parser.parse_args()
    return args
