import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import metrics
from sklearn.preprocessing import normalize


def finch(feats, finch_step, finch_dis, metric="cosine", do_normalize=True):
    if do_normalize:
        feats = normalize(feats, norm='l2').astype('float32')
    num_track = feats.shape[0]
    clusters = np.arange(num_track)

    for step in range(finch_step):
        print('Step {}'.format(step))
        pre_ids = list(set(clusters))
        pre_ids.sort()
        if step >= 3:
            print(pre_ids[-10:])
        if len(pre_ids) <= 3:
            break
        pre_map = defaultdict(list)
        for i, x in tqdm(enumerate(clusters)):
            pre_map[x].append(i)
        # if step>=3:
        #     print("pre_map before convert: ",pre_map[-10:])
        pre_map = {k: np.array(v) for k, v in pre_map.items()}

        print('Calculate center features')
        if step == 0:
            feats_now = feats.copy()
        else:
            feats_now = np.array([np.sum(feats[pre_map[i]], axis=0) / pre_map[i].size
                                  for i in tqdm(pre_ids)])
        print('Search top1')
        print("feature_shape_now: ", feats_now.shape)
        num_track_now = feats_now.shape[0]

        feats_now = normalize(feats_now, norm='l2').astype('float32')

        orig_dist = metrics.pairwise.pairwise_distances(feats_now, feats_now, metric=metric)
        np.fill_diagonal(orig_dist, float('inf'))
        topk_idx = np.argmin(orig_dist, axis=1)
        topk_scores = [orig_dist[i][topk_idx[i]] for i in range(len(orig_dist))]
        print("orig_dist: {}, topk_scores:{}".format(orig_dist, topk_scores))
        clusters_now = []
        used = [False for _ in range(num_track_now)]

        def dfs(root):
            used[root] = True
            res = set([root])
            for idx in graph[root]:
                if not used[idx]:
                    res |= dfs(idx)
            return res

        graph = [[] for i in range(num_track_now)]
        print('==Building graph==')
        for i in tqdm(range(num_track_now)):
            if finch_dis < 0 or topk_scores[i] < finch_dis:
                graph[i].append(topk_idx[i])
                graph[topk_idx[i]].append(i)

        print('DFS: ')
        for i in tqdm(range(num_track_now)):
            if used[i]:
                continue
            clusters_now.append(list(dfs(i)))

        print('Merge cluster')
        if step >= 3:
            print("what is pre_map before merge?", pre_map[0])
        new_id_cnt = len(pre_map)
        for cluster in tqdm(clusters_now):
            tmp_ids = np.array([])
            for i in cluster:
                tmp_ids = np.concatenate((tmp_ids, pre_map[i]))
                pre_map.pop(i)
            pre_map[new_id_cnt] = tmp_ids.astype('int32')
            new_id_cnt += 1

        for i, key in enumerate(pre_map.keys()):
            clusters[pre_map[key]] = i

    print('Done')
    print("final_cluster_num: {}, clusters: {}".format(len(set(clusters)), clusters))

    return clusters

