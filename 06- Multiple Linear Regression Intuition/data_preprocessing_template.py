# Data Preprocessing Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:4].values
y = data.iloc[:,-1].values


# Encoding categorical data from Labels into continious 0 1: 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

# The new method doesn't require to use LabelEncoder and you can make it by two lines
ct_X = ColumnTransformer([('3', OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct_X.fit_transform(X)

# Avoiding Dummy Variable Trap: 
X = X[:,1:]
# Splitting the dataset into training set and test set :
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = .2, random_state = 0)


# Fitting Multiple Linear Regression to the training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predecting the Test set values:
y_pred = regressor.predict(X_test)


'''Note: we can't visulaize the data because it has a lot of 
independent variabbles (features) in contrast of single linear
regression which has one feature and one output
So compare manually between y_pred & y_test ! '''


# Building the optimal model using Backward elimination: 
    
#X = np.append(arr = X , values = np.ones((50,1)).astype(int) , axis = 1 )
# a small trick to make the b0 variable at the begining of our dataset: 
X = np.append(arr =  np.ones((50,1)).astype(int)  , values = X , axis = 1 )

import statsmodels.api as sm
X_opt = np.array(X[:,[0,1,2,3,4,5]],dtype = float)
regressor_OLS = sm.OLS(endog  = y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,1,3,4,5]],dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,3,4,5]],dtype = float)
regressor_OLS = sm.OLS(endog = y, exog =X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,3,5]],dtype = float)
regressor_OLS = sm.OLS(endog = y, exog =X_opt).fit()
print(regressor_OLS.summary())

X_opt = np.array(X[:,[0,3]],dtype = float)
regressor_OLS = sm.OLS(endog = y, exog =X_opt).fit()
print(regressor_OLS.summary())

