import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = pd.read_csv("Iris.csv")
#iris.info()

X = iris.iloc[:,0:-1]
y = iris["species"]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

#print(X, "\n", y)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn.neighbors import KNeighborsClassifier
KNC = KNeighborsClassifier(n_neighbors=3)
KNC.fit(Xtrain, ytrain)
prediction = KNC.predict(Xtest)

from sklearn.metrics import confusion_matrix, classification_report, f1_score
error = confusion_matrix(ytest, prediction)
sns.heatmap(error, annot=True, fmt="d")
plt.title("Error")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print(classification_report(ytest, prediction))

#finding the best k
score = []
for i in range(1,11):
    KNC = KNeighborsClassifier(n_neighbors=i)
    KNC.fit(Xtrain, ytrain)
    prediction = KNC.predict(Xtest)
    score.append(f1_score(ytest, prediction, average="macro"))
bestValue = max(score)
index = score.index(bestValue) + 1
print(bestValue, index)
print(score)

