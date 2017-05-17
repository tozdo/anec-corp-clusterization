from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from stop_words import get_stop_words
import os

documents = [] 
stop_words = get_stop_words('russian')

path = '/home/tozdo/amystem/'
for papka in os.listdir(path):
    anum = 0
    for file in os.listdir(path + papka + '/'):
        if anum < 150:
            f = open(path + papka + '/' + file, 'r', encoding = 'utf-8')
            text = f.read()
            documents.append(text) 
            f.close()
            anum += 1
                 
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(documents)

num_k = 11 #11 персонажей 
model = KMeans(n_clusters=num_k, init='k-means++', max_iter=100)
model.fit(X)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
#fw = open('clusters.txt', 'a', encoding = 'utf-8')

for i in range(num_k):
    print("Cluster %d:" % i)
    #fw.write("Cluster %d:" % i)
    #fw.write('\n')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        #fw.write(' %s' % terms[ind] + '\n')
#fw.close()
