import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plotting')
parser.add_argument('--file_name',default='FedPAV-20200722_233845.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--fig_name',default='local-epoch-1.pdf', type=str, help='output figure path-name')
parser.add_argument('--num_epochs', default=300, type=int, help='number of epochs we want to select our metrics from, we chose 300 in the paper')

args = parser.parse_args()

with open(args.file_name, 'r') as f:
    f = f.readlines()

datasets = ['Market','DukeMTMC-reID','cuhk03-np-detected','cuhk01','MSMT17','viper','prid','3dpes','ilids']

#Due to convergence issue, We select the best 3 federated models based on Big Dataset Performance and average the Rank-1 Acc&mAP as metrics
max_metrics = [-3,-2,-1]
dic = {item:[[],[]] for item in max_metrics}

epoch_count = 0
current_local_count = 0
sum_4_datasets = 0
temp_Rank1 = []
temp_mAP = []

for line in f:
    #test frequency=10 and there are 9 datasets to test
    if epoch_count==int(args.num_epochs//10)*9:
        break
    if 'Rank' in line:
        for p in [0,1,2,4]:
            if datasets[p] in line:
                lindex = line.index(':')
                sum_4_datasets+=float(line[lindex+1:lindex+9])
        lindex = line.index(':')
        temp_Rank1.append(float(line[lindex+1:lindex+9]))
        temp_mAP.append(float(line[-8:]))
        epoch_count+=1
        current_local_count+=1
    if current_local_count==9:
        if sum_4_datasets/4>max_metrics[0]:
            del dic[max_metrics[0]]
            max_metrics[0] = sum_4_datasets/4
            max_metrics.sort()
            dic[sum_4_datasets/4] = [temp_Rank1,temp_mAP]
        sum_4_datasets=0
        current_local_count=0
        temp_Rank1=[]
        temp_mAP=[]
    
print(dic.keys())
arr = np.zeros((2,9))
for key in dic.keys():
    arr+=np.array(dic[key])
arr = arr/3
print(arr)

Rank1,mAP = arr[0]*100,arr[1]*100

plt.figure(figsize=(20,10))
name_list = ['Market\n-1501','DukeMTMC\n-reID','CUHK03\n-NP','CUHK01','MSMT\n17','VIPeR','PRID\n2011','3DPeS','iLIDS\n-VID']
index = [4,1,0,2,6,3,5,7,8]
Rank1 = [Rank1[i] for i in index]
mAP = [mAP[i] for i in index]
name_list = [name_list[i] for i in index]
x =list(range(len(index)))

total_width, n = 0.8, 2
width = total_width / n
plt.ylabel('(%)', fontsize = 35)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.bar(x, Rank1, width=width, label='Rank-1 Accuracy',fc = '#7b8b6f')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, mAP, width=width, label='mAP',tick_label = name_list,fc = '#faead3')

plt.legend(loc=3,fontsize = 35, bbox_to_anchor = (0.13,-0.3), ncol = 2)

ax = plt.gca()
ax.set_ylim([15,85])

plt.savefig(args.fig_name, bbox_inches = 'tight', dpi = 300, format='pdf')
plt.close()