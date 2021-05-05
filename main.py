# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
import time
import os
import yaml
import random
import numpy as np
import scipy.io
import pathlib
import sys
import json
import copy
import multiprocessing as mp
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from client import Client
from server import Server
from utils import set_random_seed
from data_utils import Data


mp.set_start_method('spawn', force=True)
sys.setrecursionlimit(10000)
version =  torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--project_dir',default='.', type=str, help='project path')
parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
parser.add_argument('--datasets',default='Market,DukeMTMC-reID,cuhk03-np-detected,cuhk01,MSMT17,viper,prid,3dpes,ilids,cuhk02',type=str, help='datasets used')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.5, type=float, help='drop rate')

# arguments for federated setting
parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_of_clients', default=9, type=int, help='number of clients')

# arguments for data transformation
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )

# arguments for testing federated model
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--multiple_scale',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--test_dir',default='all',type=str, help='./test_data')

parser.add_argument('--resume_epoch', default=0, type=int, help='resume from which epoch, if 0, no resume')
parser.add_argument('--experiment_index', default=0, type=int, help='index of training time')
# arguments for optimization
parser.add_argument('--cdw', action='store_true', help='use cosine distance weight for model aggregation, default false' )
parser.add_argument('--kd', action='store_true', help='apply knowledge distillation, default false')
parser.add_argument('--kd_method', default='cluster', type=str, help='whole or cluster')
parser.add_argument('--regularization', action='store_true', help='use regularization during distillation, default false' )
parser.add_argument('--clustering', action='store_true', help='use clustering to aggregate models, fault false')
parser.add_argument('--clustering_method', default='finch', type=str, help='method used for clustering, finch or kmeans')
parser.add_argument('--max_distance', default=0.9, type=float, help='maximum distance in finch algorithm')
parser.add_argument('--n_cluster', default=2, type=int, help='number of cluster in Kmeans')


def save_checkpoint(server, clients, client_list, cpk_dir, epoch):
    torch.save({
        'epoch': epoch,
        'server_state_dict': server.federated_model.state_dict(),
        'client_list': [clients[c].cid for c in client_list],
        'client_classifier': [clients[c].classifier.state_dict() for c in client_list],
        'client_model': [clients[c].model.state_dict() for c in client_list]
    }, os.path.join(cpk_dir, "{}.pth".format(epoch)))


def load_checkpoint(path):
    cpk = torch.load(path)
    epoch = cpk['epoch']
    server_state_dict = cpk['server_state_dict']
    client_list = cpk['client_list']
    client_classifier = cpk['client_classifier']
    client_model = cpk['client_model']
    return epoch, server_state_dict, client_list, client_classifier, client_model



def train():
    args = parser.parse_args()
    print(args)
    if args.clustering:
        clu = "clu"
    else:
        clu = "Nclu"

    if args.cdw:
        cdw = "cdw"
    else:
        cdw = "Ncdw"
    if args.kd:
        kd = "kd"
    else:
        kd = "Nkd"
    if args.regularization:
        reg = "reg"
    else:
        reg = "Nreg"

    kd_method = args.kd_method
    assert (kd_method == 'whole' or kd_method == 'cluster')
    if args.clustering:
        if args.clustering_method == "kmeans":
            cluster_description = "kmeans_{}".format(args.n_cluster)
        else:
            cluster_description = "finch_{}".format(args.max_distance)
    else:
        cluster_description = "No_cluster"

    cpk_dir = "checkpoints/{}_{}_{}_{}_{}_{}_{}".format(clu, cdw, kd, kd_method, reg,
                                                        cluster_description, args.experiment_index)
    cpk_dir = os.path.join(args.project_dir, cpk_dir)
    if not os.path.isdir(cpk_dir):
        os.makedirs(cpk_dir)

    epoch = args.resume_epoch

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    set_random_seed(1)

    data = Data(args.datasets, args.data_dir, args.batch_size, args.erasing_p, args.color_jitter, args.train_all)
    data.preprocess()

    clients = {}
    for cid in data.client_list:
        clients[cid] = Client(
            cid, 
            data, 
            device, 
            args.project_dir, 
            args.model_name, 
            args.local_epoch, 
            args.lr, 
            args.batch_size, 
            args.drop_rate, 
            args.stride,
            args.clustering)

    server = Server(
        clients, 
        data, 
        device, 
        args.project_dir, 
        args.model_name, 
        args.num_of_clients, 
        args.lr, 
        args.drop_rate, 
        args.stride, 
        args.multiple_scale,
        args.clustering,
        args.clustering_method,
        args.max_distance,
        args.n_cluster)

    if epoch != 0:
        print("======= loading checkpoint, epoch: {}".format(epoch))
        path = os.path.join(cpk_dir, "{}.pth".format(epoch))
        cpk_epoch, server_state_dict, client_list, client_classifier, client_model = load_checkpoint(path)
        assert (epoch == cpk_epoch)
        server.federated_model.load_state_dict(server_state_dict)
        for i in range(len(client_list)):
            cid = client_list[i]
            clients[cid].classifier.load_state_dict(client_classifier[i])
            clients[cid].model.load_state_dict(client_model[i])
        print("all models loaded, training from {}".format(epoch))

    print("=====training start!========")
    rounds = 500
    rounds = rounds // args.local_epoch
    for i in range(epoch, rounds):
        save_checkpoint(server, clients, data.client_list, cpk_dir, i)
        print('='*10)
        print("Round Number {}".format(i))
        print('='*10)
        server.train(i, args.cdw, use_cuda)
        # if not args.clustering:
        #     save_path = os.path.join(dir_name, 'federated_model.pth')
        #     torch.save(server.federated_model.cpu().state_dict(), save_path)
        if (i+1) % 10 == 0:
            server.test(use_cuda, use_fed=True)
            if args.kd:
                server.knowledge_distillation(args.regularization, kd_method)
                server.test(use_cuda, use_fed=True)
        server.draw_curve()

if __name__ == '__main__':
    train()




