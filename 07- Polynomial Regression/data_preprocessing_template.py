
# Data Preprocessing Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Position_Salaries.csv')
X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values


'''
# Splitting the dataset into training set and test set :
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = .2, random_state = 0)
'''

# Features Scaling 
"""from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)"""

# Fitting Linear Regression Model:
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)   
    
    
    
# Fitting Polynomyal Regression Model:
from sklearn.preprocessing import PolynomialFeatures 
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


# Visulaization the Linear Regression Model 
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.title('Truth or bluff (linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visulaization the Polynomial Regression Model
plt.scatter(X,y, color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visulaization the Polynomial Regression Model with better curvature:
X_grid = np.arange(min(X), max(X)+0.1, 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y)
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'orange')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Prediction a new value for a person with 6.5 level and claim that he deserves 160k$

# By Using Simple Linear Regression:
print('The person deserves by Simple Linear Regression:')
print(lin_reg.predict([[6.5]]))

# By Using Polynomial Linear Regression:
print('The person deserves by Polynomial Linear Regression:')
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))


