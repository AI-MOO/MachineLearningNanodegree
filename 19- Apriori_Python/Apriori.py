# Apriori Algorthim
'''
-------------------------------------------------------
The data set is collected for a convenience store in which the products are stored 
in columns, and rows represent the customers or visitors  of this store in which 
every row  represents a customer who bought some of these items or nor of them through
a full week. so our mission here to work on an association rule to find any corelation
 or association between these products, for the purpose of enhancing our bussines.
for example do customers are often buy Chips with Coke? 
------------------------------------------------------
    
'''

# Importing the required libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(1,7501): 
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)])
    
'''
The min_support : 3 Products are sold in 7 days through all transactions:(3*7/7500 = 0.0028 or 0.003)
The min_confidence: often we use 20%
min_left: 3 
min_length: among the dataset you want to find a corellation between at least 2 products !
Explination: the customer will buy an item will buy the associated item with a percentage of 20% 
'''
# Importing the apriori algorthim and assign the rules for the algorthim
from apyori import apriori 
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.2 , min_left = 3 , min_length =2 )

# Visuliazing the results
results = list(rules)
results_list = []

for i in range (0, len(results)):
    results_list.append('Rule:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))