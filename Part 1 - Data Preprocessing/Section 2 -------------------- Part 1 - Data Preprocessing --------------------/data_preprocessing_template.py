# My own

# Essential libraries
import numpy as np # Contains mathematical tools.
from matplotlib import pyplot as plt # Helpful for plotting stuff.
import pandas as pd # Useful when importing and managing datasets.

# Importing dataset
dataset = pd.read_csv('Data.csv')

# Matrix of features
X = dataset.iloc[:, :-1].values # "iloc[rows,columns]"
Y = dataset.iloc[:, 3].values # taking the last column

# The case of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0) # axis '0' means column.
imputer.fit(X[:, 1:3]) # Upper bound is not included -> 1:3 means 1 to 2.
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# Splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) # fit train first
X_test = sc_X.transform(X_test)

