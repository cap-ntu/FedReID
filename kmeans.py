import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

def kmeans(feats, n_clusters=2, do_normalize=True):
    if do_normalize:
        feats = normalize(feats, norm='l2').astype('float32')
    print(feats.shape)
    print(feats)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(feats)
    return kmeans.labels_