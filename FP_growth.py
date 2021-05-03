#pip install pyfpgrowth

#Importing libraries
import numpy as np
import pandas as pd

#Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#Converting the dataset values into list
transactions = []
for sublist in dataset.values.tolist():
    clean_sublist = [item for item in sublist if item is not np.nan]
    transactions.append(clean_sublist) 

#Training FP-growth on the dataset
import pyfpgrowth
patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

#Visualization the results
results = list(rules)
