from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from stop_words import get_stop_words
import os
import pandas as pd
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab

contents = [] 
pers = []
stop_words = get_stop_words('russian')

path = '/home/tozdo/amystem/'
for papka in os.listdir(path):
    pers.append(papka)
    anum = 0
    for file in os.listdir(path + papka):
        if anum < 250:
            f = open(path + papka + '/' + file, 'r', encoding = 'utf-8')
            text = f.read()
            contents.append(text) 
            f.close()
            anum += 1
                 
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(contents)
dist = 1 - cosine_similarity(X)

num_k = 11 #11 персонажей, например 
model = KMeans(n_clusters=num_k, init='k-means++', max_iter=100)
model.fit(X)
clusters = model.labels_.tolist()

MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist)                     
xs, ys = pos[:, 0], pos[:, 1]

dafr = pd.DataFrame(dict(x=xs, y=ys, label=clusters))
groups = dafr.groupby('label')
fig, ax = plt.subplots(figsize=(15,15))
ax.margins(0.2)
print(groups)
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10)

plt.show()

