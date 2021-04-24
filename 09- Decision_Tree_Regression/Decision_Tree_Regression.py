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
'''
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
'''

# Fitting The Decision Tree model with the Dataset:
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)


# Prediction a new value 
y_pred = regressor.predict([[6.5]])
print(y_pred)


# Visulaization of the Decision_Tree Model
X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y)
plt.plot(X_grid,regressor.predict(X_grid), color = 'orange')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


