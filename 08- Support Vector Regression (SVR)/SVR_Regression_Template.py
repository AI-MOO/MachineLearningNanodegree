#Regression Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values


# Splitting the dataset into training set and test set :

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = .2, random_state = 0)
'''
'''
# Features Scaling 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = np.ravel(sc_y.fit_transform(y.reshape(-1,1)))


# Fitting The SVR model with the Dataset:
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') 
regressor.fit(X,y)

# Prediction a new value 
y_pred = regressor.predict(sc_X.transform([[6.5]]))
print(y_pred)
print(sc_y.inverse_transform(y_pred))


# Visulaization of the Regression Model
plt.scatter(X,y, color = 'red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (Regression Model)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

