import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing dataset

dataset = pd.read_csv('C:\\Users\\Ozavize\\Desktop\\50_startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

table = dataset.head()
#print (table)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder =  LabelEncoder()
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer ([('State', OneHotEncoder(), [3])], remainder = 'passthrough')
X =  ct.fit_transform (X)

#X[:,3] = labelencoder.fit_transform(X[:,3])
#onehotencoder = OneHotEncoder()

#X = onehotencoder.fit_transform(X).toarray()
X = X [:, 1:] # avoiding the dummy variable trap

print (table)

#dividing the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 0)

#regressor line and fitting the line
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

print (y_pred)
