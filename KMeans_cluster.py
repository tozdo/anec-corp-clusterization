from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from stop_words import get_stop_words
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import os

contents = []
pers = []
d = {}
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
print(X)
print(dist)

#кластеризация методом KMeans
num_k = 6

model = KMeans(n_clusters=num_k, init='k-means++', max_iter=100)
model.fit(X)
clusters = model.labels_.tolist()

MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)                     
xs, ys = pos[:, 0], pos[:, 1]

df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=pers))
groups = df.groupby('label')

#визуализация
#markers = {0: 'o', 1: 'v', 2: '^', 3: 'p'}
fig, ax = plt.subplots(figsize=(30,40))
ax.margins(0.2)
font = {'family': 'Droid Sans',
        'weight': 'normal'}
rc('font', **font)

for name, group in groups:
    ax.plot(group.x, group.y, marker='.', linestyle='',
            ms=15)

for i in range(len(df)):
    ax.text(df.ix[i]['x'],df.ix[i]['y'], df.ix[i]['title'], size=15)

plt.savefig('1cluster.png', dpi=200)

