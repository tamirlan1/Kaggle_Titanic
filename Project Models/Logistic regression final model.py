# -*- coding: utf-8 -*-
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
from sklearn.grid_search import GridSearchCV
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

#use settings from tuned classifier to set up a moe

logistic = linear_model.LogisticRegression(class_weight ='balanced')
logistic = logistic.fit(X,y)
y_predicted_train= logistic.predict_proba(X)#predicted class for training set

#obtain critical threshold
maxrev=0
final_threshold =0.5
for x in xrange(1,100):
    thresh = 0.01*x
    predicted_y_train=np.array([1 if x > thresh else 0 for x in list(y_predicted_train[:,1])])
    cmatrix=confusion_matrix(y, predicted_y_train)
    newROI=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)
    if newROI>maxrev:
        maxrev=newROI
        final_threshold=thresh

##get predictions on test set and use the optimal threshold determined on the training set to determine classes

y_predicted_test= logistic.predict_proba(X_test)#predicted class for test set
predicted_y_test=np.array([1 if x > final_threshold else 0 for x in list(y_predicted_test[:,1])])

##obtain relevant statistics

cm =ConfusionMatrix(y_test, predicted_y_test)
cm.print_stats()
acc=accuracy_score(y_test, predicted_y_test) 
cmatrix=confusion_matrix(y_test, predicted_y_test)
ROI=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)

   

#
