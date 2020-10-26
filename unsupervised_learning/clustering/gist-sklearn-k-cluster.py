import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Sci-kit learn one liner
x,y = make_blobs(n_samples=100, 
                 centers=3, 
                 n_features=2, 
                 random_state=0)

colours = {0:'#1F77B4', 
           1:'#D62728', 
           2:'#2CA02C'}

kmeans = KMeans(n_clusters=3, random_state=0).fit(x)

for idx, i in enumerate(x):
    plt.scatter(i[0], i[1], c=colours[kmeans.labels_[idx]], alpha=0.5)

for i in kmeans.cluster_centers_:
    plt.scatter(i[0],i[1], c='#000000')

plt.axis('off')
plt.show()