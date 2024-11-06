import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("HousingData.csv")
dataframe.info()
print(dataframe.isnull().sum())
dataframe = dataframe.dropna()

X = dataframe[["RM", "LSTAT"]]
y = dataframe["MEDV"]

Xtrain, Xtest, ytrain, ytest = train_test_split(X , y, test_size=0.3, random_state=3)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(Xtrain, ytrain)
prediction = model.predict(Xtest)
print(prediction, "\n")
print(ytest)

from sklearn.metrics import root_mean_squared_error as rmse
print(rmse(ytest, prediction))

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
PolyFeatures = PolynomialFeatures(2)
XtrainPoly = PolyFeatures.fit_transform(Xtrain)
print(XtrainPoly)
XtestPoly = PolyFeatures.fit_transform(Xtest)

model.fit(XtrainPoly, ytrain)
prediction = model.predict(XtestPoly)
print(rmse(ytest, prediction))



