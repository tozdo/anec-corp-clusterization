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
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
clusters = db.labels_.tolist()

print(Counter(labels))

MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)                     
xs, ys = pos[:, 0], pos[:, 1]

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=pers))
groups = df.groupby('label')

#визуализация

font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)

plt.figure(1, figsize=(25, 25))

for name, group in groups:
    if name == -1:
        plt.plot(group.x, group.y, 'o', color='k', markersize=6)
    else:
        plt.plot(group.x, group.y, 'o', markersize=18)


#plt.title('Estimated number of clusters: %d' % n_clusters_)

for i in range(len(df)):
    if df.ix[i]['label'] != -1: #не помечаем шум
        plt.text(df.ix[i]['x'],df.ix[i]['y'], df.ix[i]['title'], size=15)

#plt.show()
plt.savefig('DBSCAN_cluster.png', dpi=200)
