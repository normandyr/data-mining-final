# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 22:54:32 2019

@author: Normandy Bryson
CSC535 Data Mining - Final Project
Evaluating Classification Methods
"""

'''import os
dirpath = os.getcwd()
print("current directory is : " + dirpath)

os.chdir("\\Users\\A\\Desktop")

dirpath = os.getcwd()
print("current directory is : " + dirpath)'''

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#sns.set_style("white")

#f, ax = plt.subplots(figsize=(30, 30))

#here we have our usable dataset with X corresponding to the attributes and y corresponding to the class outcome
data = pd.read_csv('transfusion.csv')
X = np.array(data[["Months", "Times", "Blood", "Frequency"]])
y = np.array(data["Donated"])

'''this shows count of class outcomes
sns.countplot(x='Donated', data=data)'''

'''using shuffle isn't allowed because no sparse matrix but sklearn will let us do this anyways without warning us if you use this will lower accuracy'''
#seed = 42
#X, y = shuffle(X, y, random_state=seed)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state=0)

"""using GaussianNB because attributes are continuous"""
clf = GaussianNB()
clf.fit(X_train, y_train)
print(clf.class_prior_)
y_pred = clf.predict(X_test)


#print("Accuracy:", accuracy_score(y_test, y_pred))
'''will use confusion matrix in report'''
#print(confusion_matrix(y_test, y_pred))
print("Naive Bayes ")
print(classification_report(y_test, y_pred))
print("______________________________________")


"""using logistic regression because the outcome variable is binary 0 or 1"""
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(accuracy_score(y_test, y_pred))
print("Logistic ")
print(classification_report(y_test, y_pred))
print("______________________________________")




"""K-nearest neighbors with continous attributes"""
knn = KNeighborsClassifier(n_neighbors=9)
scores = cross_val_score(knn, X, y, cv= 5)
print("knn Cross-validation scores: ", scores)
print("knn Cross-validation mean scores: ", scores.mean())

knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, y_train)
print("this is KNN ", knn.score(X_test, y_test))
'''
Best k is k= 9
for k in range(1, 13):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20)
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    print("this is KNN " ,k,  knn.score(X_test, y_test))'''



"""support vector machines"""
X = StandardScaler().fit_transform(X)
clf = SVC(kernel="linear",gamma="auto")
clf.fit(X, y)
y_pred = clf.predict(X_test)
print("SVM ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("______________________________________")






