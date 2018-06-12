##imports

import numpy as np
import random
import csv
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from pandas_confusion import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from pandas_confusion import ConfusionMatrix
from sklearn.metrics import confusion_matrix


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

#set up classifier
neigh = KNeighborsClassifier(n_neighbors=35, weights='distance')#algorithm="auto",n_jobs=-1,

neigh = neigh.fit(X,y)
y_predicted_train= neigh.predict_proba(X)#predicted class for training set

#obtain optimal probability threshold for classification
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
    
y_predicted_test= neigh.predict_proba(X_test)#predicted probability for test set
predicted_y_test=np.array([1 if x > final_threshold else 0 for x in list(y_predicted_test[:,1])])#apply threshold to classify the test set

#obtain relevant statistics
cm =ConfusionMatrix(y_test, predicted_y_test)
cm.print_stats()
acc=accuracy_score(y_test, predicted_y_test) 
cmatrix=confusion_matrix(y_test, predicted_y_test)
ROI=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)

