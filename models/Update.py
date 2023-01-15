#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from models.Nets import fix_bn
import copy
import utils.util as util

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label




class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, worker_id=0, logger=None):
        self.args = args
        self.loss_func = torch.nn.CrossEntropyLoss()  # nn.BCEWithLogitsLoss()  torch.nn.MSELoss() #

        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs
        self.worker_id = worker_id
        self.train_iter = self.ldr_train.__iter__()
        #记录当前状态
        self.epoch = 0
        self.step = 0
        self.local_state_dict = {}
        self.logger = logger
        self.offset = {}
        self.local_list = ['fc2', 're4', 'fc3'] #['classifier'] #['fc3'] #['conv1', 'b1', 're1', 'fc3']   #['conv1', 'b1', 're1']

        self.gamma = 1.0

    def get_next_batch(self):
        try:
            batch_data = self.train_iter.next()
        except:
            #每个epoch重新划分训练集和测试集（这里测试集实际应该是验证集）
            self.ldr_train = DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=self.args.local_bs, shuffle=True)
            self.train_iter = self.ldr_train.__iter__()
            batch_data = self.train_iter.next()
            self.epoch += 1
        self.step += 1

        return batch_data

    def update_local_model(self, global_model):
        global_model.update(self.local_state_dict)
        return global_model

    def penalty0(self, model: nn.Module, count: float):
        loss = torch.tensor(0.0)
        for n, p in model.named_parameters():
            _loss = count * (p - self._means[n]) ** 2
            loss = loss + _loss.sum()
        return loss




    def penalty5(self, model: nn.Module, w_1order_glob:dict, b: torch.tensor, count: float):
        #loss = torch.tensor(0.0)
        c = torch.tensor(0.0)
        a = torch.tensor(0.0)
        for n, p in model.named_parameters():
            if n in w_1order_glob.keys():
                a_v = p - self._means[n]
                _a =  a_v ** 2
                _c = (a_v - w_1order_glob[n]) ** 2
                a = a + _a.sum()
                c = c + _c.sum()

        if b>0:
            trilist = [torch.sqrt(a).detach().cpu().numpy(), torch.sqrt(b).detach().cpu().numpy(), torch.sqrt(c).detach().cpu().numpy()]
            if trilist[0]  + trilist[1] < trilist[2]:
                print('a={:.6f}, b={:.6f}, c={:.6f}.'.format(trilist[0], trilist[1], trilist[2]))
            e = torch.tensor(1e-4)
            loss = count * (1-(a+b-c)/(2*torch.sqrt((a*b+e))))
        else:
            loss = torch.tensor(0.0)
        return loss


    def get_batchnorm_parameters(self, net):
        dic = {}
        for module_prefix, module in net.named_modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                dic[module_prefix] = copy.deepcopy(module.state_dict())
        return dic


    def set_para(self, weights):
        self._means = {}
        for n,p in weights.items():
            self._means[n] = p.clone().detach()

    def set_rangen(self, seed):
        '''
        初始化随机数生成器
        :return:
        '''
        torch.manual_seed(seed)



    def set_mask(self, weights):
        self.mask = {}
        for n,p in weights:
            self.mask[n] = (torch.rand(p.size()) < 0.5).to(p.device)

    def set_offset(self, weights):
        self.offset = {}
        for n,p in weights:
            self.mask[n] = (torch.rand(p.size()) < 0.5).to(p.device)


    def update_mask(self, change_mask):
        for n, p in self.mask.items():
            self.mask[n] = torch.bitwise_xor(p, change_mask[n])

    def update_direction(self):
        self.direction = {}
        b = torch.tensor(0.0)
        for n, p in self.mask.items():
            self.direction[n] = self.mask[n].float()
            _b = self.direction[n] ** 2
            b = b + _b.sum()
        self.direction_len = b

    def inverse_mask(self):
        for n, p in self.mask.items():
            self.mask[n] = ~self.mask[n]

    def train(self, net, step_in_round, global_round, lr, args, change_mask=None, w_1order_glob={}):

        net.train()
        #if global_round > 0:
        #    net.apply(fix_bn)

        self.set_para(net.state_dict())



        b = torch.tensor(0.0)
        for k, v in net.named_parameters():
            if k in w_1order_glob.keys():
                _b = w_1order_glob[k] ** 2
                b = b + _b.sum()




        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr) #, momentum=self.args.momentum)

        correct = 0
        total = 0
        batch_loss = []
        loss1 = torch.tensor(0.0)
        for iter_step in range(step_in_round):
            images, targets = self.get_next_batch()
            images = images.to(self.args.device)
            targets = targets.to(self.args.device)
            labels = targets


            net.zero_grad()


            log_probs = net(images)
            loss0 = self.loss_func(log_probs, labels) #+ nn.CrossEntropyLoss()(log_probs, targets)
            _, predicted = log_probs.max(1)

            if args.method == 1 :
                loss1 = self.penalty0(net, args.imp0)
                loss = loss0 + loss1
            elif args.method == 7:
                loss1 = self.penalty5(net, w_1order_glob, b, args.imp0) #+ self.penalty0(net, args.imp0)
                loss = loss0 + loss1
            else:
                #loss1 = self.penalty2(net, 0.1)
                loss = loss0 #+ loss1

            loss.backward()
            optimizer.step()


            correct += predicted.eq(targets).sum().item()
            total += labels.size(0)
            batch_loss.append(loss.item())

            #util.log_batchnorm_record_one(net, self.worker_id, self.logger, self.get_step())

        # 计算移动距离
        dis = torch.sqrt(self.penalty0(net, 1.0).detach()).cpu().numpy()

        print('Worker: {}: Update Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Loss1: {:.6f}, acc: {:.2f}%({}/{}, dis: {:.6f})'.format(self.worker_id,
            self.epoch, self.get_past_batch(), self.get_step_per_epoch(),
                  100. * self.get_past_batch() / self.get_step_per_epoch(), loss.item(), loss1.item(), 100. * correct / total, correct, total, dis))

        if self.worker_id < 2:
            self.logger.add_scalar('local_dis'+str(self.worker_id), dis, global_round)




        return net.state_dict(), None, sum(batch_loss) / len(batch_loss), correct/total, dis, self.offset


    def get_step_per_epoch(self):
        '''
        获取当前epoch中第几个step
        :return:
        '''
        return len(self.ldr_train)

    def get_epoch(self):
        '''
        获取当前所处的epoch
        :return:
        '''
        return self.epoch

    def get_step(self):
        '''
        获取当前总的step数量
        :return:
        '''
        return self.step

    def get_past_batch(self):
        '''
        get the used batch in the current epoch
        :return:
        '''
        return  self.step - self.epoch*len(self.ldr_train)

    def get_remain_batch(self):
        '''
        get the remain batch in the current epoch
        :return:
        '''
        return len(self.ldr_train)-self.get_past_batch()


