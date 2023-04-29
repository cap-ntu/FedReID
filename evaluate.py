import scipy.io
import torch
import numpy as np
import os
# import argparse
# parser = argparse.ArgumentParser(description='Training')
# parser.add_argument('--result_dir', default='.', type=str)
# parser.add_argument('--dataset', default='no_dataset', type=str)
# args = parser.parse_args()

#######################################################################
# Evaluate
def evaluate(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def testing_model(result, dataset):
    # result = scipy.io.loadmat(file_path)
    # print("========= after loading ==========")
    # for i in result:
    #     print(i, np.array(result[i]).shape)

    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = np.array(result['query_cam'])
    query_label = np.array(result['query_label'])
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = np.array(result['gallery_cam'])
    gallery_label = np.array(result['gallery_label'])
    # print(type(query_feature),query_feature[:3])
    # print(type(query_cam),query_cam[:3])
    # print(type(query_label),query_label[:3])

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    print(query_feature.shape)
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print(dataset+' Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0], CMC[4], CMC[9], ap/len(query_label)))
    print('-'*15)
    print()
