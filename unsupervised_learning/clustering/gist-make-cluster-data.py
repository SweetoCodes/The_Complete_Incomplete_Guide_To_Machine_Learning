import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

Creating and visualising the data
x,y = make_blobs(n_samples=100, 
                 centers=3, 
                 n_features=2, 
                 random_state=0)

colours = {0:'#1F77B4', 
           1:'#D62728', 
           2:'#2CA02C'}

for idx, i in enumerate(x):
    plt.scatter(i[0], i[1], c=colours[y[idx]])

plt.axis('off')
plt.show()