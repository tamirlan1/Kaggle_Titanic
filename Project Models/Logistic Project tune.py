# -*- coding: utf-8 -*-
##modified syntax from various Logistic regression and hyperparameter search pieces and merged and modified them to create a tuner for our logistic model
#imports
import numpy as np
from sklearn import linear_model
from sklearn import linear_model, datasets, metrics
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pandas_confusion import ConfusionMatrix
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV

path = "C:/Users/Lorna/Documents/R/MSE_235"

##Import Training data created on R
fname = "NewsDataTrain.csv"
full_file = path + "/" + fname

cols = range(62)
data_train = np.loadtxt(full_file,delimiter=",", skiprows=1, usecols=tuple(cols[1:]))
X = data_train[:,1:-2]
y = data_train[:,-1]

##Import Test data created on R
gname = "NewsDataTest.csv"
full_file_test = path + "/" + gname

data_test = np.loadtxt(full_file_test,delimiter=",", skiprows=1, usecols=tuple(cols[1:]))
X_test = data_test[:,1:-2]
y_test = data_test[:,-1]

#I set up a model with prior weights
logistic = linear_model.LogisticRegression(class_weight ='balanced')
logistic = logistic.fit(X,y)
y_predicted = logistic.predict(X_test)#predicted class
cm =ConfusionMatrix(y_test, y_predicted)
cm.print_stats()
acc=accuracy_score(y_test, y_predicted) 
#print(classification_report(y_test, y_predicted))
cmatrix=confusion_matrix(y_test, y_predicted)
ROI=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)

#I then tried regularization (Using grid search to search for the most optimal regularization parameter c)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
GridSearchCV(cv=None,
       estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
          penalty='l2', tol=0.0001),
       param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
clf = clf.fit(X,y)
y_predicted2 = clf.predict(X_test)#predicted class
cm2 =ConfusionMatrix(y_test, y_predicted2)
cm2.print_stats()
acc2=accuracy_score(y_test, y_predicted2) 
#print(classification_report(y_test, y_predicted))
cmatrix2=confusion_matrix(y_test, y_predicted)
ROI2=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)

##Next I tried cross validation
logisticCV = linear_model.LogisticRegressionCV(class_weight ='balanced',scoring='roc_auc')#scoring =‘accuracy’
logisticCV = logisticCV.fit(X,y)
y_predicted = logisticCV.predict(X_test)#predicted class
cm =ConfusionMatrix(y_test, y_predicted)
cm.print_stats()
acc=accuracy_score(y_test, y_predicted) 
#print(classification_report(y_test, y_predicted))
cmatrix=confusion_matrix(y_test, y_predicted)
ROI=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)

