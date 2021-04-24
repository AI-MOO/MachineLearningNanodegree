# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data from Labels into continious 0 1: 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer


# Encoding categorical data in 'Country' Column to continious 0 1:
# The new method doesn't require to use LabelEncoder and you can make it by two lines
ct_X = ColumnTransformer([('0', OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct_X.fit_transform(X)

# Encoding categorical data in 'Gender' Column to continious 0 1:
LabelEncoder_X = LabelEncoder()
X [:,4] = LabelEncoder_X.fit_transform(X[:,4])

# Remove one of 3 dummy variables for the country 
X = np.array(transform.fit_transform(X), dtype = np.float64)
X = X[:,1:]

# Splitting the dataset into training set and test set :
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state = 0)



# Fitting XGBoost version 0.90 to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies_mean = accuracies.mean()
accuracies_std = accuracies.std()
