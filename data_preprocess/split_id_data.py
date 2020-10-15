import os 
import numpy as np
import shutil
import random

os.chdir('data')
os.mkdir('cuhk03_byid_2')
os.mkdir('cuhk03_byid_2/train_all')
os.mkdir('cuhk03_byid_2/val')

for dataset in ['cuhk03-np-detected']:
    os.chdir(dataset+'/pytorch')

    all_data = os.listdir('train_all')
    random.shuffle(all_data)
    num_data = len(all_data)
    num_parts = 2
    len_parts = num_data//num_parts

    for i in range(num_parts):
        destination = '../../cuhk03_byid_2/train_all/'+dataset+'_part'+str(i+1) 
        os.mkdir(destination)
        for folder in all_data[i*len_parts:(i+1)*len_parts]:
            if folder[0]=='.':
                continue
            shutil.copytree('train_all/'+folder, destination+'/'+folder)
    os.chdir('../../')

for dataset in ['cuhk03-np-detected']:
    os.chdir(dataset+'/pytorch')

    all_data = os.listdir('val')
    random.shuffle(all_data)
    num_data = len(all_data)
    num_parts = 3
    len_parts = num_data//num_parts

    for i in range(num_parts):
        destination = '../../cuhk03_byid_2/val/'+dataset+'_part'+str(i+1) 
        os.mkdir(destination)
        for folder in all_data[i*len_parts:(i+1)*len_parts]:
            if folder[0]=='.':
                continue
            shutil.copytree('val/'+folder, destination+'/'+folder)
    os.chdir('../../')
