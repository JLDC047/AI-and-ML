import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

titantic = pd.read_csv("titanic.csv")
#titantic.info()

y = titantic["Survived"]
X = titantic[["Pclass", "Sex", "Age", "Siblings/Spouses Aboard", "Parents/Children Aboard", "Fare"]]
#print(X.head(10))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])
#print(X.head(10))

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, random_state=3)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(Xtrain, ytrain)
predicted = model.predict(Xtest)
#print(predicted[0:11])
#print(ytest.head(10))

from sklearn.metrics import confusion_matrix
error = confusion_matrix(ytest, predicted)
print(error)

sns.heatmap(error, annot=True, fmt="d")
plt.title("Error")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(ytest, predicted))


