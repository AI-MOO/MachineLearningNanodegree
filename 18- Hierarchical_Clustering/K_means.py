# Data Preprocessing Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:,[3,4]].values
#y = data.iloc[:,-1].values


# Using the elbow method to find the optimal nunmber of clusters 
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', random_state = 42 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()




'''
# Splitting the dataset into training set and test set :
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.25, random_state = 0)

# Features Scaling 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)
'''



# Fitting the Decision Tree Classification model to training dataset: 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


