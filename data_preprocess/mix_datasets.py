import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--data_dir',default='data',type=str, help='training dir path')
opt = parser.parse_args()

data_dir = opt.data_dir

dest_dir = os.path.join(data_dir, 'fused_all')
if not os.path.isdir(dest_dir):
    os.mkdir(dest_dir)


datasets = ['Market', 'DukeMTMC-reID', 'cuhk03-np-detected', 'cuhk01', 'MSMT17', 'viper', 'prid', '3dpes', 'ilids']

for dataset in datasets:
    path = os.path.join(data_dir, dataset, 'pytorch')

    folders = os.listdir(path)

    for f in folders:
        if 'camera' in f:
            continue
        dest_folder = os.path.join(dest_dir, f)
        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)

        folder_path = os.path.join(path, f)

        identities = os.listdir(folder_path)
        for id in identities:
            shutil.copytree(os.path.join(folder_path, id), os.path.join(dest_folder, id))
