# Importing the libraries: 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset 
data = pd.read_csv('Salary_Data.csv')
# Train set must be matrix because that I performed [:,0:1]
X = data.iloc[:,0:1].values
y = data.iloc[:,1].values

# Splitting the dataset into training set and test set :
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 1/3 , random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the Test set result: 
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train) 

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary in USD $')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'orange')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Experience in years')
plt.ylabel('Salary in USD $')
plt.show()

