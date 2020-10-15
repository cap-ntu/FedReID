import torch
from torchvision import datasets
import os 
import numpy as np
import torchvision.datasets.folder
import shutil

for dataset in ['MSMT17']:
    os.chdir('data/'+dataset+'/pytorch')
    os.mkdir('camera_clients')

    dataset_train = datasets.ImageFolder('train_all')
    cameras = np.array(list(map(lambda x: x.split('c')[-1].split('_')[0], list(zip(*dataset_train.imgs))[0])))
    os.mkdir('camera_clients/train_all')
    for cam in sorted(set(cameras)):
        os.mkdir('camera_clients/train_all/'+str(cam))
    for i in range(len(dataset_train)):
        file,label = dataset_train.imgs[i]
        cam = cameras[i]
        data_path = 'camera_clients/train_all/'+str(cam)+'/'+str(label)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        shutil.copyfile(file, data_path+'/'+file.split('/')[-1])

    dataset_val = datasets.ImageFolder('val')
    print(dataset_val)
    cameras = np.array(list(map(lambda x: x.split('c')[-1].split('_')[0], list(zip(*dataset_val.imgs))[0])))
    os.mkdir('camera_clients/val')
    for cam in sorted(set(cameras)):
        os.mkdir('camera_clients/val/'+str(cam))
    for i in range(len(dataset_val)):
        file,label = dataset_val.imgs[i]
        cam = cameras[i]
        data_path = 'camera_clients/val/'+str(cam)+'/'+str(label)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        shutil.copyfile(file, data_path+'/'+file.split('/')[-1])
