from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from stop_words import get_stop_words
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.datasets.samples_generator import make_blobs

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

stop_words = get_stop_words('russian')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(contents)
dist = 1 - cosine_similarity(X)

#кластеризация
db = DBSCAN(eps=0.9, min_samples=4).fit(X)
labels = db.labels_
clusters = db.labels_.tolist()

elincl = Counter(clusters)

MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)                     
xs, ys = pos[:, 0], pos[:, 1]

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=pers))
groups = df.groupby('label')
df2 = pd.DataFrame(dict(label=clusters, title=pers))
groups2 = df2.groupby('label')
counters = []
for name, group in groups2:
    if name != -1:
        counters.append(Counter(group.title).most_common())
print(counters)
#визуализация

font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)
centers = [[-11, -9], [-2, -9], [5, -9], [-11, 2], [-2, 2], [5, 2]]

plt.figure(1, figsize=(20, 15))


for name, group in groups:
    print(name)
    #counters.append(Counter(group.title))
    if name == -1:
        plt.plot(group.x, group.y, 'o', color='k', markersize=6)
        #continue
    else:
        X0, y = make_blobs(n_samples=elincl[name], centers=centers[name],
                           n_features=2)
        xss = X0[:, 0]
        yss = X0[:, 1]
        plt.plot(xss, yss, 'o', markersize=18, label=counters[name])
        #plt.text(group.x, group.y, group['title'], size=14)
plt.legend(numpoints=1, loc=2, ncol=2, bbox_to_anchor=(0., 1.02, 1., .102),
           fontsize=15, borderaxespad=0.)
#plt.title('Estimated number of clusters: %d' % n_clusters_)

#for i in range(len(df)):
#     if df.ix[i]['label'] != -1: #не помечаем шум
#         plt.text(df.ix[i]['x'],df.ix[i]['y'], df.ix[i]['title'], size=15)

#plt.show()
plt.savefig('DBSCAN_cluster_all.png', dpi=200)
