# Artificial Neural Network   

# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13].values
y = data.iloc[:,13].values

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
X = X[:,1:]

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

# Importing Tensorflow + Keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

# Inintialization the ANN 
classifier = Sequential()

# Adding the input layer and the first hidden layer  
# Output = units = 11 + 1 / 2 , kernel_initializer = weights random uniform distribution , activation = relu ('rectifier')  
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the input layer and the first hidden layer  
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu',))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )


#Fitting the ANN to the training set 
classifier.fit(X_train, y_train, batch_size = 10 , epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


