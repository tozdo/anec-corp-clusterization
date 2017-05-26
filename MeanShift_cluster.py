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
from collections import Counter

contents = []
pers = []
path = '/home/tozdo/amystem/'
for papka in os.listdir(path):
    for file in os.listdir(path + papka):
        f = open(path + papka + '/' + file, 'r', encoding='utf-8')
        text = f.read()
        contents.append(text)
        pers.append(papka)
        f.close()

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
df2 = pd.DataFrame(dict(label=labels, title=pers))
groups = df2.groupby('label')
counters = []
for name, group in groups:
    counters.append(Counter(group.title).most_common())
#визуализация
plt.figure(1, figsize=(35, 15))
plt.clf()
font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(pos[my_members, 0], pos[my_members, 1], col + '.', markersize=20, label=counters[k])
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=25)

#for i in range(n_clusters_):
    #plt.plot(label=counters[i])
    
plt.title('Estimated number of clusters: %d' % n_clusters_, weight='normal', loc='right')

plt.legend(numpoints=1, loc=2, ncol=2, bbox_to_anchor=(0., 1.02, 1., .102),
           fontsize=13, borderaxespad=0.)

#for i in range(len(df)):
#    plt.text(df.ix[i]['x'],df.ix[i]['y'], df.ix[i]['title'], size=18)
#plt.show()
plt.savefig('MeanShiftcluster_all.png', dpi=200)
