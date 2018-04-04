"""
     PULSAR DISTANCE MEASUREMENT - MULTIPLE LINEAR REGRESSION MODEL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data import
dataset = pd.read_csv('Pulsars.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
   

# Data split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

"""****     BACKWARD ELIMINATION    ****"""
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

"""Optimal matrix of features (optimal team of significant variables i.e.
   having high impact on dependent variable 'y', SL = 0.05"""
X_opt = X[:, [0, 1, 2, 3, 4]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Optimal matrix without variable with highest P-value

X_opt = X[:, [0, 2, 3, 4]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# Optimal matrix without variable with highest P-value

X_opt = X[:, [2, 3, 4]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# FINAL MODEL WITH P < SL = 0.05 FOR x1 and x2

X_opt = X[:, [2, 4]]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

"""***      BACKWARD ELIMINATION    ***"""




