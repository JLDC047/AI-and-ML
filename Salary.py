import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

salary = pd.read_csv("Salary.csv")
X = salary["YearsExperience"]
y = salary["Salary"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LinearRegression
model1= LinearRegression()
Xtrain = np.array(Xtrain).reshape(-1, 1)
model1.fit(Xtrain, ytrain)
m = model1.coef_
c = model1.intercept_

array = np.arange(1,15)
p = m * array + c
plt.scatter(X, y)
plt.plot(array, p, c="r")
plt.show()

Xtest = np.array(Xtest).reshape(-1, 1)
ypredicted = model1.predict(Xtest)
print(ypredicted)

from sklearn.metrics import root_mean_squared_error as rmse
print(rmse(ytest, ypredicted))

