import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plotting')
parser.add_argument('--file_name',default='FedPAV-20200722_233845.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--fig_name',default='local-epoch-1-curve.pdf', type=str, help='output figure path-name')
parser.add_argument('--num_epochs', default=300, type=int, help='number of epochs we want to plot, we chose 300 in the paper')
parser.add_argument('--dataset', default='Market,DukeMTMC-reID,cuhk03-np-detected,MSMT17',type=str, help='The datasets we want to plot')

args = parser.parse_args()

with open(args.file_name, 'r') as f:
    f = f.readlines()

datasets = args.dataset.split(',')
acc = {x:[] for x in datasets}
mAP = {x:[] for x in datasets}

epoch_count = 0
local_count = 0
for line in f:
    if epoch_count==int(args.num_epochs//10)*len(datasets):
        break
    if ('Rank' in line) and (line.split(' ')[0] in datasets):
        name = line.split(' ')[0]
        index = line.index(':')
        acc[name].append(float(line[index+1:index+9]))
        mAP[name].append(float(line[-8:]))

        local_count+=1
        epoch_count+=1
        if local_count==len(datasets):
            local_count = 0

length = len(list(acc.values())[0])

plt.figure()
for name in acc.keys():
    plt.plot(np.arange(1,length+1)*10, acc[name], label = name)
plt.xlabel('Epochs')
plt.ylabel('Rank-1 Accuracy (%)')
plt.xlim(0,(length+1)*10)
plt.legend(loc=3)
plt.savefig('_ACC.'.join(args.fig_name.split('.')), dpi = 300)
plt.close()

plt.figure()
for name in mAP.keys():
    plt.plot(np.arange(1,length+1)*10, mAP[name], label = name)
plt.xlabel('Epochs')
plt.ylabel('mAP (%)')
plt.xlim(0,(length+1)*10)
plt.legend(loc=3)
plt.savefig('_mAP.'.join(args.fig_name.split('.')), dpi = 300)
plt.close()