# Data Preprocessing Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:,[3,4]].values
#y = data.iloc[:,-1].values


# Finding the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch 
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

# Visulaization the dendrogram method
plt.title('dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eculidean Distance')
plt.show()

# Using the the Hierarchical Clustering by AgglomerativeClustering 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5 , affinity = 'euclidean', linkage = 'ward')
y_pred = hc.fit_predict(X)

# Clusters Visulizations
plt.scatter(X[y_pred == 0,0],X[y_pred == 0,1], s = 100 , c = 'red', label = 'Careful' )
plt.scatter(X[y_pred == 1,0],X[y_pred == 1,1], s = 100 , c = 'blue', label = 'Standard' )
plt.scatter(X[y_pred == 2,0],X[y_pred == 2,1], s = 100 , c = 'green', label = 'Target' )
plt.scatter(X[y_pred == 3,0],X[y_pred == 3,1], s = 100 , c = 'orange', label = 'Careless' )
plt.scatter(X[y_pred == 4,0],X[y_pred == 4,1], s = 100 , c = 'yellow', label = 'Sensible' )
plt.title('Hierarchical Clustering: ')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
