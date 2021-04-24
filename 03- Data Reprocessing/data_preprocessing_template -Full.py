
# Data Preprocessing Template: 

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Data.csv')
X = data.iloc[:,:3].values
y = data.iloc[:,-1].values


# Scikit learn library: handling missing values 
from sklearn.impute import SimpleImputer
# strategy can be: mean , median , most_frequent
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


# Encoding categorical data from Labels into continious 0 1: 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

# The new method doesn't require to use LabelEncoder and you can make it by two lines
ct_X = ColumnTransformer([('0', OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct_X.fit_transform(X)

LabelEncoder_y = LabelEncoder()
y= LabelEncoder_y.fit_transform(y)


# Splitting the dataset into training set and test set :
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = .2, random_state = 0)

# Features Scaling 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)