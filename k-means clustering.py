# K-MEANS CLUSTERING
##The main difference between K-Means and K-Medoids is that K-Means will form clusters based on the distance of observations to each centroid, while K-Medoid forms clusters based on the distance to medoids.
### to look at results of inferencing this algorithm, refer to https://github.com/Torajabu/Standard-ML-Algorithms---Python/tree/main/k%20means%20clustering%20reults
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x1 = 10*np.random.rand(100,2)
x1.shape

kmean = KMeans(n_clusters=3)
kmean.fit(x1)
kmean.cluster_centers_
kmean.labels_

wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x1)
    wcss.append(kmeans.inertia_)
    print('Cluster', i, 'Inertia', kmeans.inertia_)

plt.plot(range(1,20), wcss)
plt.title('The Elbow Curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # WCSS stands for total within-cluster sum of square
plt.show()
