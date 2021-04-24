# Natural Language Processing 

# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the text
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

cropus = []
for i in range(0,1000):
    # remove any spechial character and replace it by spance 
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    # convert all uppercase letters to lowercase letters  
    review = review.lower()
    # spliting the words in a list 
    review = review.split()
    # get root of words such as loves or loving converted to love 
    ps = PorterStemmer()
    
    # we keep the most importat words and we remove unnecessary words such as [this, the ,and .. ]
    # List Comprehensions procedure:   
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # combine all words in one string 
    review = ' ' .join(review)
    cropus.append(review)

# Creating the bag of words 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(cropus).toarray()
y = dataset.iloc[:,1]

# Splitting the dataset into training set and test set :
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.25, random_state = 0)


# Fitting the Decision Tree Classification model to training dataset: 
from sklearn.ensemble import RandomForestClassifier  
classifier = RandomForestClassifier(n_estimators = 10 ,criterion = 'entropy',random_state = 0 )
classifier.fit(X_train, y_train)

# predecting the test set results 
y_pred = classifier.predict(X_test)

# Creating the confusion matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)



