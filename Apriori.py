
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv" , header = None)
 #Converting the dataset values into list
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)])
    
#Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions,min_support = 0.003,min_confidence = 0.2,min_lift =3,min_lenght = 2)

#Visualization the results
results = list(rules)
