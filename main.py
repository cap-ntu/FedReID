# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import multiprocessing as mp
import os
import sys
import time

import matplotlib
import torch

from client import Client
from data_utils import Data
from server import Server
from utils import set_random_seed

matplotlib.use('agg')
mp.set_start_method('spawn', force=True)
sys.setrecursionlimit(10000)
version = torch.__version__

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--model_name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--project_dir', default='.', type=str, help='project path')
parser.add_argument('--data_dir', default='data', type=str, help='training dir path')
parser.add_argument('--datasets',
                    default='Market,DukeMTMC-reID,cuhk03-np-detected,cuhk01,MSMT17,viper,prid,3dpes,ilids', type=str,
                    help='datasets used')
parser.add_argument('--train_all', action='store_true', default=True, help='use all training data')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--drop_rate', default=0.5, type=float, help='drop rate')

# arguments for federated setting
parser.add_argument('--local_epoch', default=1, type=int, help='number of local epochs')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_of_clients', default=9, type=int, help='number of clients')
parser.add_argument('--rounds', default=300, type=int, help='training rounds')

# arguments for data transformation
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')

# arguments for testing federated model
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--multiple_scale', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

# arguments for optimization
parser.add_argument('--cdw', action='store_true',
                    help='use cosine distance weight for model aggregation, default false')
parser.add_argument('--kd', action='store_true', help='apply knowledge distillation, default false')
parser.add_argument('--regularization', action='store_true',
                    help='use regularization during distillation, default false')


def train():
    args = parser.parse_args()
    print(args)

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
            args.stride)

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
        args.multiple_scale)

    save_path = os.path.join(args.project_dir, 'model')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, "{}_{}".format(args.model_name, args.rounds))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print("=====training start!========")
    for i in range(args.rounds):
        print('=' * 10)
        print("Round Number {}".format(i))
        print('=' * 10)
        server.train(i, args.cdw, use_cuda)
        save_path = os.path.join(save_path, 'federated_model.pth')
        torch.save(server.federated_model.cpu().state_dict(), save_path)
        if (i + 1) % 10 == 0:
            if args.kd:
                server.knowledge_distillation(args.regularization)
            server.test(use_cuda, save_path)
        server.draw_curve()


if __name__ == '__main__':
    train()
