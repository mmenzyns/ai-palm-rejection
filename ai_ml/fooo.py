from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


import joblib
import pandas as pd


X_train, y_train = joblib.load("train_data.joblib")
X_test, y_test = joblib.load("test_data.joblib")

print(X_train.shape, y_train.shape)

X_train = pd.concat((X_train[:1000], X_train[-1000:]))
y_train = pd.concat((y_train[:1000], y_train[-1000:]))

print(X_train)
print(X_train.shape, y_train.shape)

clf = SVC(verbose=True)
clf.fit(X_train, y_train)

scores = cross_val_score(clf, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))

joblib.dump(clf, "SVM_temp.joblib")
scores = cross_val_score(clf, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))


clf = None
clf = joblib.load("SVM_temp.joblib")

scores = cross_val_score(clf, X_test, y_test, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))