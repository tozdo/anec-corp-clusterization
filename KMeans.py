from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from stop_words import get_stop_words
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os
from collections import Counter
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler

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

#кластеризация методом KMeans
num_k = 11

model = KMeans(n_clusters=num_k, init='k-means++', max_iter=100)
model.fit(X)
cluster_centers = model.cluster_centers_
flat = cluster_centers.flatten(0)
labels = model.labels_
clusters = model.labels_.tolist()

elincl = Counter(clusters)

df = pd.DataFrame(dict(label=clusters, title=pers))
groups = df.groupby('label')
counters = []
for name, group in groups:
    counters.append(Counter(group.title).most_common())


plt.figure(1, figsize=(10, 5))
font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)
centers = [[-15, -13], [-10, -13], [-5, -13], [0, -13], [5, -13], [10, -13],
           [-12, 2], [-7, 2], [-2, 2], [3, 2], [7, 2]]
colors = {0:'red', 1:'blue', 2:'green', 3:'violet', 4:'black',
          5:'pink', 6:'yellow', 7:'magenta', 8:'tan', 9:'cyan', 10: 'lime'}
for i in range(num_k):
    X0, y = make_blobs(n_samples=elincl[i],centers=centers[i], n_features=2) #centers=centers[i]
    xs = X0[:, 0]
    ys = X0[:, 1]
    m = i - 1
    plt.plot(xs, ys,'o', label=counters[i], c=colors[i]) 
    
plt.legend(numpoints=1, loc=2, ncol=2, bbox_to_anchor=(0., 1.02, 1., .102),
           fontsize=5, borderaxespad=0.)

#plt.show()
plt.savefig('11_clustersKM_all.png', dpi=200)


