import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plotting')
parser.add_argument('--file_name1',default='FINAL_script_local1-20200414_105220.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--file_name2',default='FINAL_script_local5-20200414_105613.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--file_name3',default='FINAL_script_local10-20200414_105643.log', type=str,help='path-name of the log file to be read')
parser.add_argument('--fig_name',default='local-epochs.pdf', type=str, help='output figure path-name')
parser.add_argument('--num_epochs', default=300, type=int, help='number of epochs we want to select our metrics from, we chose 300 in the paper')

args = parser.parse_args()
files = []
for name in [args.file_name1, args.file_name2, args.file_name3]:
    with open(name, 'r') as f:
        f = f.readlines()
        files.append(f)

datasets = ['Market','DukeMTMC-reID','cuhk03-np-detected','cuhk01','MSMT17','viper','prid','3dpes','ilids']

acc_list = []

for f in files:
    #Due to convergence issue, We select the best 3 federated models based on Big Dataset Performance and average the Rank-1 Acc&mAP as metrics
    max_metrics = [-3,-2,-1]
    dic = {item:[] for item in max_metrics}

    epoch_count = 0
    current_local_count = 0
    sum_4_datasets = 0
    temp_Rank1 = []

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
            epoch_count+=1
            current_local_count+=1
        if current_local_count==9:
            if sum_4_datasets/4>max_metrics[0]:
                del dic[max_metrics[0]]
                max_metrics[0] = sum_4_datasets/4
                max_metrics.sort()
                dic[sum_4_datasets/4] = temp_Rank1
            sum_4_datasets=0
            current_local_count=0
            temp_Rank1=[]
        
    arr = np.zeros(9)
    for key in dic.keys():
        arr+=np.array(dic[key])
    arr = arr/3*100
    acc_list.append(list(arr))

print(acc_list)

plt.figure(figsize=(20,10))
name_list = ['Market\n-1501','DukeMTMC\n-reID','CUHK03\n-NP','CUHK01','MSMT\n17','VIPeR','PRID\n2011','3DPeS','iLIDS\n-VID']
index = [4,1,0,2,6,3,5,7,8]
for i in range(len(acc_list)):
    acc_list[i] = [acc_list[i][j] for j in index]
name_list = [name_list[i] for i in index]
x =list(range(len(index)))

total_width, n = 0.8, 3
width = total_width / n
plt.ylabel('Rank-1 Accuracy (%)', fontsize = 35)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.bar(x, acc_list[0], width=width, label='E=1',fc = '#7b8b6f')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, acc_list[1], width=width, label='E=5',fc = '#faead3')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, acc_list[2], width=width, label='E=10',tick_label = name_list,fc = '#965454')

plt.legend(loc=3,fontsize = 35, bbox_to_anchor = (0.13,-0.3), ncol = 3)

ax = plt.gca()
ax.set_ylim([15,85])

plt.savefig(args.fig_name, bbox_inches = 'tight', dpi = 300, format='pdf')
plt.close()