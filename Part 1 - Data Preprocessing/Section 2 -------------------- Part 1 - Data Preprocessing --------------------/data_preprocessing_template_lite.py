# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 21:58:43 2018

@author: faraz

Following is a template that will be used before creating every model.
"""

# Essential libraries
import numpy as np # Contains mathematical tools.
from matplotlib import pyplot as plt # Helpful for plotting stuff.
import pandas as pd # Useful when importing and managing datasets.

from sklearn.preprocessing import StandardScaler
# Importing dataset
dataset = pd.read_csv('Data.csv') # Change name of the dataset.

# Matrix of features
X = dataset.iloc[:, :-1].values # "iloc[rows,columns]"
Y = dataset.iloc[:, 3].values # taking the last column

# Missing data code here (not always required so removed)
# Enoding categorical data code here (not always required so removed)

# Splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Feature scaling
""" Since most ML libraries include feature scaling, this is left here just in case.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # fit train first
X_test = sc_X.transform(X_test)
"""
