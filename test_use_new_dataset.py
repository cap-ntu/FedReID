from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms
import os
import json
import torch
from random_erasing import RandomErasing
from utils import get_model, extract_feature
from evaluate import testing_model

class ImageDataset(Dataset):
    def __init__(self, imgs,  transform = None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data, label = self.imgs[index]
        return self.transform(Image.open(data)), label


dataset = "CAVIARa"
test_dir = os.path.join("/mnt/lustre/ganxin1/FedReID", "CAVIARa", "pytorch")
gallery_dataset = datasets.ImageFolder(os.path.join(test_dir, 'gallery'))
query_dataset = datasets.ImageFolder(os.path.join(test_dir, 'query'))


transform_val = [
                transforms.Resize(size=(256,128),interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
t = transforms.Compose(transform_val)

gallery_dataset = ImageDataset(gallery_dataset.imgs, t)
query_dataset = ImageDataset(query_dataset.imgs, t)

test_loaders = {}
test_loaders[dataset] = {key: torch.utils.data.DataLoader(
                                                dataset, 
                                                batch_size=32,
                                                shuffle=False, 
                                                num_workers=8, 
                                                pin_memory=True) for key, dataset in {'gallery': gallery_dataset, 'query': query_dataset}.items()}
        


def load_checkpoint(path):
    cpk = torch.load(path)
    epoch = cpk['epoch']
    server_state_dict = cpk['server_state_dict']
    client_list = cpk['client_list']
    client_classifier = cpk['client_classifier']
    client_model = cpk['client_model']
    return epoch, server_state_dict, client_list, client_classifier, client_model

import torch.nn as nn
model = get_model(750, 0.5, 2)
model.classifier.classifier = nn.Sequential()
model.eval()
cpk_path  = "/mnt/lustre/ganxin1/FedReID/checkpoints/clu_cdw_Nkd_Nreg/299.pth"
cpk_epoch, server_state_dict, client_list, client_classifier, client_model = load_checkpoint(cpk_path)

model.load_state_dict(server_state_dict)

from torch.autograd import Variable
def fliplr(img):
    """flip horizontal
    """
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_feature(model, dataloaders, ms):
    features = torch.FloatTensor()
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        ff = torch.FloatTensor(n, 512).zero_() # cuda

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img) # cuda
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    print(input_img)
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

with torch.no_grad():
    gallery_feature = extract_feature(model, test_loaders[dataset]['gallery'], [1])
    query_feature = extract_feature(model, test_loaders[dataset]['query'], [1])

def get_camera_ids(img_paths):
    """get camera id and labels by image path
    """
    camera_ids = []
    labels = []
    camera = 0 if 'gallery' in img_paths else 1
    for path, v in img_paths:
        filename = os.path.basename(path)
        
        label = filename[0:4]

        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        
        camera_ids.append(camera)
    return camera_ids, labels
gallery_meta = {}
query_meta = {}
gallery_cameras, gallery_labels = get_camera_ids(gallery_dataset.imgs)
gallery_meta[dataset] = {
    'sizes':  len(gallery_dataset),
    'cameras': gallery_cameras,
    'labels': gallery_labels
}

query_cameras, query_labels = get_camera_ids(query_dataset.imgs)
query_meta[dataset] = {
    'sizes':  len(query_dataset),
    'cameras': query_cameras,
    'labels': query_labels
}

result = {
    'gallery_f': gallery_feature.numpy(),
    'gallery_label': data.gallery_meta[dataset]['labels'],
    'gallery_cam': data.gallery_meta[dataset]['cameras'],
    'query_f': query_feature.numpy(),
    'query_label': data.query_meta[dataset]['labels'],
    'query_cam': data.query_meta[dataset]['cameras']
}


print(dataset)
testing_model(result, dataset)
