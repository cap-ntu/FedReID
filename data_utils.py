from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import json
import torch
from random_erasing import RandomErasing

class ImageDataset(Dataset):
    def __init__(self, imgs,  transform = None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data,label = self.imgs[index]
        return self.transform(Image.open(data)), label


class Data():
    def __init__(self, datasets, data_dir, batch_size, erasing_p, color_jitter, train_all):
        self.datasets = datasets.split(',')
        self.batch_size = batch_size
        self.erasing_p = erasing_p
        self.color_jitter = color_jitter
        self.data_dir = data_dir
        self.train_all = '_all' if train_all else ''
        
    def transform(self):
        transform_train = [
                transforms.Resize((256,128), interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop((256,128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        transform_val = [
                transforms.Resize(size=(256,128),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]

        if self.erasing_p > 0:
            transform_train = transform_train + [RandomErasing(probability=self.erasing_p, mean=[0.0, 0.0, 0.0])]

        if self.color_jitter:
            transform_train = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train

        self.data_transforms = {
            'train': transforms.Compose(transform_train),
            'val': transforms.Compose(transform_val),
        }        

    def preprocess_kd_data(self, dataset):
        loader, image_dataset = self.preprocess_one_train_dataset(dataset)
        self.kd_loader = loader


    def preprocess_one_train_dataset(self, dataset):
        """preprocess a training dataset, construct a data loader.
        """
        data_path = os.path.join(self.data_dir, dataset, 'pytorch')
        data_path = os.path.join(data_path, 'train' + self.train_all)
        image_dataset = datasets.ImageFolder(data_path)

        loader = torch.utils.data.DataLoader(
            ImageDataset(image_dataset.imgs, self.data_transforms['train']), 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=2, 
            pin_memory=False)

        return loader, image_dataset

    def preprocess_train(self):
        """preprocess training data, constructing train loaders
        """
        self.train_loaders = {}
        self.train_dataset_sizes = {}
        self.train_class_sizes = {}
        self.client_list = []
        
        for dataset in self.datasets:
            self.client_list.append(dataset)
          
            loader, image_dataset = self.preprocess_one_train_dataset(dataset)

            self.train_dataset_sizes[dataset] = len(image_dataset)
            self.train_class_sizes[dataset] = len(image_dataset.classes)
            self.train_loaders[dataset] = loader
            
        print('Train dataset sizes:', self.train_dataset_sizes)
        print('Train class sizes:', self.train_class_sizes)
        
    def preprocess_test(self):
        """preprocess testing data, constructing test loaders
        """
        self.test_loaders = {}
        self.gallery_meta = {}
        self.query_meta = {}

        for test_dir in self.datasets:
            test_dir = 'data/'+test_dir+'/pytorch'

            dataset = test_dir.split('/')[1]
            gallery_dataset = datasets.ImageFolder(os.path.join(test_dir, 'gallery'))
            query_dataset = datasets.ImageFolder(os.path.join(test_dir, 'query'))
    
            gallery_dataset = ImageDataset(gallery_dataset.imgs, self.data_transforms['val'])
            query_dataset = ImageDataset(query_dataset.imgs, self.data_transforms['val'])

            self.test_loaders[dataset] = {key: torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=self.batch_size,
                                                shuffle=False, 
                                                num_workers=8, 
                                                pin_memory=True) for key, dataset in {'gallery': gallery_dataset, 'query': query_dataset}.items()}
        

            gallery_cameras, gallery_labels = get_camera_ids(gallery_dataset.imgs)
            self.gallery_meta[dataset] = {
                'sizes':  len(gallery_dataset),
                'cameras': gallery_cameras,
                'labels': gallery_labels
            }

            query_cameras, query_labels = get_camera_ids(query_dataset.imgs)
            self.query_meta[dataset] = {
                'sizes':  len(query_dataset),
                'cameras': query_cameras,
                'labels': query_labels
            }

        print('Query Sizes:', self.query_meta[dataset]['sizes'])
        print('Gallery Sizes:', self.gallery_meta[dataset]['sizes'])

    def preprocess(self):
        self.transform()
        self.preprocess_train()
        self.preprocess_test()
        self.preprocess_kd_data('cuhk02')

def get_camera_ids(img_paths):
    """get camera id and labels by image path
    """
    camera_ids = []
    labels = []
    for path, v in img_paths:
        filename = os.path.basename(path)
        if filename[:3]!='cam':
            label = filename[0:4]
            camera = filename.split('c')[1]
            camera = camera.split('s')[0]
        else:
            label = filename.split('_')[2]
            camera = filename.split('_')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_ids.append(int(camera[0]))
    return camera_ids, labels
