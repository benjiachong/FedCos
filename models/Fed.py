#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        if w_avg[k].numel() != 1:
            w_avg[k] = torch.div(w_avg[k], len(w))
        else:
            w_avg[k] = w_avg[k] // len(w)
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def FedWeightedAvg(w, p):
    p1 = np.array(p)/p[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + w[i][k] * p1[i]
        if w_avg[k].numel() != 1:
            w_avg[k] = torch.div(w_avg[k], sum(p1))
        else:
            w_avg[k] = w_avg[k] // sum(p1)
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg









def server_opt(w_glob_old, w_1order_glob, server_momentum, momentum=0):
    if momentum > 0:
        for k, v in w_1order_glob.items():
            if k in server_momentum:
                server_momentum[k] = momentum * server_momentum[k] + (1 - momentum) * w_1order_glob[k]
            else:
                server_momentum[k] = w_1order_glob[k]
    else:
        server_momentum = w_1order_glob

    rate = 1
    w_glob = {}
    for k, v in w_glob_old.items():
        w_glob[k] = v + rate * server_momentum[k]

    return w_glob