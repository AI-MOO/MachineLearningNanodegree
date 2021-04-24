#Regression Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Data.csv')
X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values


# Splitting the dataset into training set and test set :
'''
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = .2, random_state = 0)
'''

# Features Scaling 
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)"""

# Fitting The Regression model with the Dataset:

    

# Prediction a new value 
print(regressor.predict([[6.5]]))


# Visulaization of the Regression Model
plt.scatter(X,y, color = 'red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (Regression Model)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y)
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'orange')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Better Visulaization of the Regression Model 
X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,y)
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'orange')

plt.title('Truth or bluff (Regression Model)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()