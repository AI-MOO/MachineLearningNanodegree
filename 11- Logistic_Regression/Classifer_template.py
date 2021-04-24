# Data Preprocessing Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:,2:4].values
y = data.iloc[:,-1].values


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




# Fitting the classifier model to training dataset: 





# Predection Process: 
y_pred = 
    


# Confusion Matrix:
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visulaization the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train , y_train
x1, x2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1 , stop = X_set[:,0].max() + 1 , step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1 , stop = X_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j , 1],
                c = ListedColormap(('red','green')) (i), label = j) 
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()   


# Visulaization the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test , y_test
x1, x2 = np.meshgrid(np.arange(start = X_set[:,0].min() - 1 , stop = X_set[:,0].max() + 1 , step = 0.01),
                     np.arange(start = X_set[:,1].min() - 1 , stop = X_set[:,1].max() + 1 , step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75 , cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j , 1],
                c = ListedColormap(('red','green')) (i), label = j) 
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()   