import os
import math
import random
import matplotlib.pyplot as plt
from utils import get_model, extract_feature
import torch.nn as nn
import torch
import scipy.io
import copy
import numpy as np
from data_utils import ImageDataset
import torch.optim as optim
from torchvision import datasets
# from finch import FINCH
from finch_dis import finch
from kmeans import kmeans
from evaluate import testing_model

def add_model(dst_model, src_model, dst_no_data, src_no_data):
    if dst_model is None:
        result = copy.deepcopy(src_model)
        return result
    params1 = src_model.named_parameters()
    params2 = dst_model.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].set_(param1.data*src_no_data + dict_params2[name1].data*dst_no_data)
    return dst_model

def scale_model(model, scale):
    params = model.named_parameters()
    dict_params = dict(params)
    with torch.no_grad():
        for name, param in dict_params.items():
            dict_params[name].set_(dict_params[name].data * scale)
    return model

def aggregate_models(models, weights):
    """aggregate models based on weights
    params:
        models: model updates from clients
        weights: weights for each model, e.g. by data sizes or cosine distance of features
    """
    if models == []:
        return None
    model = add_model(None, models[0], 0, weights[0])
    total_no_data = weights[0]
    for i in range(1, len(models)):
        model = add_model(model, models[i], total_no_data, weights[i])
        model = scale_model(model, 1.0 / (total_no_data+weights[i]))
        total_no_data = total_no_data + weights[i]
    return model


class Server():
    def __init__(self, clients, data, device, project_dir, model_name, num_of_clients, lr,
                 drop_rate, stride, multiple_scale, clustering=False, clustering_method="finch",
                 max_distance=2, n_cluster=2):
        self.project_dir = project_dir
        self.data = data
        self.device = device
        self.model_name = model_name
        self.clients = clients
        self.client_list = self.data.client_list
        self.num_of_clients = num_of_clients
        self.lr = lr
        self.multiple_scale = multiple_scale
        self.drop_rate = drop_rate
        self.stride = stride

        self.multiple_scale = []
        for s in multiple_scale.split(','):
            self.multiple_scale.append(math.sqrt(float(s)))

        self.federated_model = get_model(750, drop_rate, stride).to(device)
        self.federated_model.classifier.classifier = nn.Sequential()
        self.federated_model.eval()
        self.train_loss = []
        self.use_clustering = clustering
        self.clustering_group_for_kd = None
        self.cdw = None
        self.clients_using = None
        self.clients_weights = None
        self.clustering_method = clustering_method
        self.max_dis = max_distance
        self.n_cluster = n_cluster



    def train(self, epoch, cdw, use_cuda):
        self.cdw = cdw
        loss = []
        if not self.use_clustering:
            models = []
            cos_distance_weights = []
        data_sizes = []
        current_client_list = random.sample(self.client_list, self.num_of_clients)
        self.clients_using = current_client_list
        feature_lists = []
        for i in current_client_list:
            if not self.use_clustering:
                self.clients[i].train(self.federated_model, use_cuda)
                cos_distance_weights.append(self.clients[i].get_cos_distance_weight())
                models.append(self.clients[i].get_model())
            else:
                self.clients[i].train(None, use_cuda=use_cuda)
            loss.append(self.clients[i].get_train_loss())
            data_sizes.append(self.clients[i].get_data_sizes())

        if (epoch + 1) % 10 == 0:
            print("before aggregation, local testing")
            self.test(use_cuda)
        if self.use_clustering:
            for _, (inputs, targets) in enumerate(self.data.train_loaders['cuhk02']):
                inputs, target = inputs.to(self.device), targets.to(self.device)
                break
            for i in current_client_list:
                feature_lists.append(self.clients[i].generate_custom_data_feature(inputs).cpu().detach().numpy())
            feature_lists = np.array(feature_lists)
            # c, num_clust, _ = FINCH(feature_lists, min_sim=self.max_dis)
            # self.clustering_method, self.max_dis, self.n_cluster = n_cluster
            if self.clustering_method == "kmeans":
                clusters = kmeans(feature_lists, n_clusters=self.n_cluster, do_normalize=True)
            else:
                clusters = finch(feature_lists, finch_step=1, finch_dis=self.max_dis, metric="cosine", do_normalize=True)
            id_groups = self.clustering(clusters, current_client_list)
            print("id_groups", id_groups)


        if epoch==0:
            self.L0 = torch.Tensor(loss) 

        avg_loss = sum(loss) / self.num_of_clients

        print("==============================")
        # print("number of clients used:", len(models))
        print('Train Epoch: {}, AVG Train Loss among clients of lost epoch: {:.6f}'.format(epoch, avg_loss))
        print()

        self.train_loss.append(avg_loss)
        if self.use_clustering:
            for i in id_groups.keys():
                models = []
                data_num = []
                cos_distance = []
                for j in id_groups[i]:
                    models.append(self.clients[j].get_model())
                    data_num.append(self.clients[j].get_data_sizes())
                    cos_distance.append(self.clients[j].get_cos_distance_weight())
                if cdw:
                    federated_model = aggregate_models(models, cos_distance)
                else:
                    federated_model = aggregate_models(models, data_num)
                for j in id_groups[i]:
                    self.clients[j].set_model(federated_model)
            print("using clustering, client models set")
            self.clustering_group_for_kd = id_groups

        else:
            weights = data_sizes
            if cdw:
                print("cos distance weights:", cos_distance_weights)
                weights = cos_distance_weights
            self.federated_model = aggregate_models(models, weights)


    def draw_curve(self):
        plt.figure()
        x_epoch = list(range(len(self.train_loss)))
        plt.plot(x_epoch, self.train_loss, 'bo-', label='train')
        plt.legend()
        dir_name = os.path.join(self.project_dir, 'model', self.model_name)
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        plt.savefig(os.path.join(dir_name, 'train.png'))
        plt.close('all')
        
    def test(self, use_cuda, use_fed=False):
        print("="*10)
        print("Start Testing！")
        print("="*10)
        print('We use the scale: %s' % self.multiple_scale)
        
        for dataset in self.data.datasets:
            # if self.use_clustering:
            if use_fed and not self.use_clustering:
                print("Using federated model")
                client_model = self.federated_model.eval()
                if use_cuda:
                    client_model = self.federated_model.cuda()
            else:
                print("Using local model")
                client_model = self.clients[dataset].get_model().eval()  # self.federated_model.eval()
                if use_cuda:
                    client_model = client_model.cuda()  # self.federated_model.cuda()
            # else:
            #     self.federated_model = self.federated_model.eval()
            #     if use_cuda:
            #         self.federated_model = self.federated_model.cuda()

            with torch.no_grad():
                gallery_feature = extract_feature(client_model, self.data.test_loaders[dataset]['gallery'], self.multiple_scale)
                query_feature = extract_feature(client_model, self.data.test_loaders[dataset]['query'], self.multiple_scale)

            result = {
                'gallery_f': gallery_feature.numpy(),
                'gallery_label': self.data.gallery_meta[dataset]['labels'],
                'gallery_cam': self.data.gallery_meta[dataset]['cameras'],
                'query_f': query_feature.numpy(),
                'query_label': self.data.query_meta[dataset]['labels'],
                'query_cam': self.data.query_meta[dataset]['cameras']}
            file_path = os.path.join(self.project_dir,
                                     'model',
                                     self.model_name,
                                     'pytorch_result_{}_{}.mat'.format(dataset, random.randint(0, 100000000)))
            scipy.io.savemat(file_path, result)
                        
            print(self.model_name)
            print(dataset)
            testing_model(file_path, dataset)
            # os.system('python evaluate.py --result_dir {} --dataset {}'.format(os.path.join(self.project_dir, 'model', self.model_name), dataset))

    def knowledge_distillation(self, regularization, kd_method):
        if self.use_clustering and kd_method == 'cluster':
            print("personlaization with kd_method cluster")
            for i in self.clustering_group_for_kd:
                print("grouping {} for kd".format(self.clustering_group_for_kd[i]))
                first_client_id = self.clustering_group_for_kd[i][0]
                model = self.clients[first_client_id].get_model()
                federated_model = self.cluster_knowledge_distillation(model,
                                                                      self.clustering_group_for_kd[i],
                                                                      regularization)
                for j in self.clustering_group_for_kd[i]:
                    self.clients[j].set_model(federated_model)
        elif self.use_clustering and kd_method == 'whole':
            print("personlaization with kd_method whole")
            models = []
            cos_distance_weights = []
            data_sizes = []
            for i in self.clients_using:
                cos_distance_weights.append(self.clients[i].get_cos_distance_weight())
                models.append(self.clients[i].get_model())
                data_sizes.append(self.clients[i].get_data_sizes())
            weights = data_sizes
            if self.cdw:
                print("cos distance weights:", cos_distance_weights)
                weights = cos_distance_weights
            self.federated_model = aggregate_models(models, weights)

            federated_model = self.cluster_knowledge_distillation(self.federated_model,
                                                                  self.client_list,
                                                                  regularization)
            for j in self.client_list:
                self.clients[j].set_model(federated_model)
        else:
            MSEloss = nn.MSELoss().to(self.device)
            optimizer = optim.SGD(self.federated_model.parameters(), lr=self.lr*0.01, weight_decay=5e-4, momentum=0.9, nesterov=True)
            self.federated_model.train()

            for _, (x, target) in enumerate(self.data.kd_loader):
                x, target = x.to(self.device), target.to(self.device)
                # target=target.long()
                optimizer.zero_grad()
                soft_target = torch.Tensor([[0]*512]*len(x)).to(self.device)

                for i in self.client_list:
                    i_label = (self.clients[i].generate_soft_label(x, regularization))
                    soft_target += i_label
                soft_target /= len(self.client_list)

                output = self.federated_model(x)

                loss = MSEloss(output, soft_target)
                loss.backward()
                optimizer.step()
                print("train_loss_fine_tuning", loss.data)

    def cluster_knowledge_distillation(self, model, c_list, regularization):
        MSEloss = nn.MSELoss().to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.lr * 0.01, weight_decay=5e-4, momentum=0.9,
                              nesterov=True)
        model.train()

        for _, (x, target) in enumerate(self.data.kd_loader):
            x, target = x.to(self.device), target.to(self.device)
            # target=target.long()
            optimizer.zero_grad()
            soft_target = torch.Tensor([[0] * 512] * len(x)).to(self.device)

            for i in c_list:
                i_label = (self.clients[i].generate_soft_label(x, regularization))
                soft_target += i_label
            soft_target /= len(self.client_list)

            output = model(x)

            loss = MSEloss(output, soft_target)
            loss.backward()
            optimizer.step()
            print("train_loss_fine_tuning of {} is {}".format(c_list, loss.data))
        return model

    def clustering(self, indexs, client_list):
        id_groups = {}  # dict.fromkeys([i for i in range(num_of_cluster[0])],[])
        assert len(indexs) == len(client_list)
        for i in range(len(client_list)):
            if indexs[i] not in id_groups.keys():
                id_groups[indexs[i]] = [client_list[i]]
            else:
                id_groups[indexs[i]].append(client_list[i])
        return id_groups



