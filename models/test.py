#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import utils.util as util

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data_loader = DataLoader(datatest, batch_size=args.bs)
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

from scipy import stats

def test_ensemble(net, clients, datatest, args, current_epoch):
    # testing
    w_glob = net.state_dict()

    num_classes = 10
    all_data_record = np.array([])
    test_loss = 0
    correct = 0
    correct_single = [0]*args.num_users
    with torch.no_grad():
        data_loader = DataLoader(datatest, batch_size=args.bs)
        l = len(data_loader)
        for idata, (data, target) in enumerate(data_loader):
            predict = []
            predict_entropy = []
            y_pred = []
            confidence_list = []

            one_batch_record = np.array([])


            for idx in range(args.num_users):
                local = clients[idx]
                w_glob = local.update_local_model(w_glob)
                net.load_state_dict(w_glob)

                net.eval()

                if args.gpu != -1:
                    data, target = data.to(args.device), target.to(args.device)
                log_probs = net(data)
                #log_probs = torch.sigmoid(log_probs)
                # sum up batch loss

                if args.uncertainty != 0:
                    evidence = util.relu_evidence(log_probs)
                    _, y_p = torch.max(log_probs, 1)
                    alpha = evidence + 1
                    confidence = num_classes / torch.sum(alpha, dim=1)
                    p = (alpha / torch.sum(alpha, dim=1, keepdim=True)).cpu().numpy()
                    predict.append(p)

                else:
                    p = F.softmax(log_probs, dim=1).cpu().numpy()    #选项内对选择的确信度
                    predict.append(p)

                    #pentropy = stats.entropy(p, axis=1)
                    #predict_entropy.append(pentropy)

                    # get the index of the max log-probability
                    #confidence, _ = log_probs.max(1)  #对选择的置信度（结果是否在自己范围内）
                    #y_p = [i for item in log_probs.data.max(1, keepdim=True)[1].cpu().numpy() for i in item]
                    confidence, y_p = log_probs.data.max(1)
                    confidence = 1 - confidence

                confidence = confidence.cpu().numpy()
                y_p = y_p.cpu().numpy()
                y_pred.append(y_p)
                confidence_list.append(confidence)

                if current_epoch == 96 and current_epoch % 4 == 0:
                    if one_batch_record.size != 0:
                        one_batch_record = np.concatenate((one_batch_record, log_probs.cpu().numpy()), axis=1)
                    else:
                        one_batch_record = log_probs.cpu().numpy()
                    #one_batch_record = np.concatenate((one_batch_record, pentropy.reshape(-1,1)), axis=1)
                    one_batch_record = np.concatenate((one_batch_record, confidence.reshape(-1, 1)), axis=1)
                    one_batch_record = np.concatenate((one_batch_record, np.array(y_p).reshape(-1,1)), axis=1)

                y_single_pred = log_probs.data.max(1, keepdim=True)[1]
                correct_single[idx] = correct_single[idx] + y_single_pred.eq(target.data.view_as(y_single_pred)).long().cpu().sum()

                print('===> Saving models...')
                state = {
                    'state': net.state_dict(),
                    'epoch': current_epoch  # 将epoch一并保存
                }
                torch.save(state, './lenet'+str(idx)+'.t7')

            #client_choice = [np.argmin(z) for z in np.array(predict_entropy).T]
            client_choice = [np.argmin(z) for z in np.array(confidence_list).T]
            #client_choice = [random_pick(range(len(z)), 1.0/np.array(z)) for z in np.array(predict_entropy).T]
            y_ensemble_pred = [np.array(y_pred)[enum, item] for item, enum in enumerate(client_choice)]

            correct += sum(np.array(y_ensemble_pred) == (target.data).cpu().numpy())

            if current_epoch == 96 and current_epoch % 4 == 0:
                one_batch_record = np.concatenate((one_batch_record, np.array(y_ensemble_pred).reshape(-1,1)), axis=1)
                one_batch_record = np.concatenate((one_batch_record, (target.data).cpu().numpy().reshape(-1,1)), axis=1)

                if all_data_record.size != 0:
                    all_data_record = np.concatenate((all_data_record, one_batch_record), axis=0)
                else:
                    all_data_record = one_batch_record



    if current_epoch == 96 and current_epoch % 4 == 0:
        pd.DataFrame(all_data_record).to_csv('all_data_record_round'+str(current_epoch)+'.csv')
    return 100.00 * correct / len(data_loader.dataset), 0, [100.00 * x / len(data_loader.dataset) for x in correct_single]





def random_pick(some_list,probabilities):
    x=random.uniform(0,1)
    ps = np.array(probabilities) ** 2
    ps = ps/sum(ps)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,ps):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            break
    return item
