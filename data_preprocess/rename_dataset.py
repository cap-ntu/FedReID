import os
import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir',default='data/3dpes',type=str, help='training dir path')
parser.add_argument('--postfix',default='3d',type=str, help='dataset postfix')
opt = parser.parse_args()

data_dir = opt.data_dir
postfix = opt.postfix

path = os.path.join(data_dir, 'pytorch')
folders = os.listdir(path)

for f in folders:
    folder_path = os.path.join(path, f)
    identities = os.listdir(folder_path)
    for id in identities:
        id_path = os.path.join(folder_path, id)
        os.rename(id_path, id_path+'-'+postfix)
