#pip install mlxtend 
#pip install --no-binary :all: mlxtend

import numpy as np
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv',names=np.arange(1,21))

# Preprocesing data to fit model
transactions = []
for sublist in dataset.values.tolist():
    clean_sublist = [item for item in sublist if item is not np.nan]
    transactions.append(clean_sublist)



from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_x = pd.DataFrame(te_ary, columns=te.columns_) # encode to onehot

# Train model using Apiori algorithm 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
df_sets = apriori(df_x, min_support=0.005, use_colnames=True)
df_rules = association_rules(df_sets,metric='support',min_threshold= 0.005,support_only=True)#eclat (support) used here

#Visualization the results
results = list(df_rules)

