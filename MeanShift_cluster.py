import os
import pandas as pd
import numpy as np
from stop_words import get_stop_words
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import rc

contents = []
pers = []
path = '/home/tozdo/amystem/'
for papka in os.listdir(path):
    im = 0 
    for file in os.listdir(path + papka):
        if im < 150: 
            f = open(path + papka + '/' + file, 'r', encoding='utf-8')
            text = f.read()
            contents.append(text)
            pers.append(papka)
            f.close()
            im += 1 

#векторизация

stop_words = get_stop_words('russian')
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(contents)

dist = 1 - cosine_similarity(X)

MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)
xs, ys = pos[:, 0], pos[:, 1]
#кластеризация методом MeanShift

bandwidth = estimate_bandwidth(pos, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(pos)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print(n_clusters_)
df = pd.DataFrame(dict(x=xs, y=ys, label=labels, title=pers))
#визуализация
plt.figure(1, figsize=(25, 25))
plt.clf()
font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(pos[my_members, 0], pos[my_members, 1], col + '.', markersize=20)
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=25)
plt.title('Estimated number of clusters: %d' % n_clusters_, weight='normal')
for i in range(len(df)):
    plt.text(df.ix[i]['x'],df.ix[i]['y'], df.ix[i]['title'], size=18)
#plt.show()
plt.savefig('3cluster150.png', dpi=200)
