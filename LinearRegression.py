import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,6)
y = np.array([1,3,2,4,3])

#find the gradient
meanX = np.mean(x)
meanY = np.mean(y)

m = np.sum((x - meanX) * (y - meanY)) / np.sum((x - meanX) ** 2)
c = meanY - m * meanX

print(m, c)
p = m * x + c

plt.scatter(x, y)
plt.plot(x, p)
plt.show()

#estimating error RSME 
error = np.sqrt(np.sum((p - y) ** 2) / len(y))
print(error)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
X = x.reshape(-1, 1)
LR.fit(X, y)
print(LR.coef_)
print(LR.intercept_)
prediction = LR.predict([[102]])
print(prediction)

