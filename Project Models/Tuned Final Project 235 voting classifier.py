##imports

import numpy as np
import random
import csv
from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from pandas_confusion import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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

##Setting up classifiers
clf1 = LogisticRegression(class_weight='balanced')
clf2 = RandomForestClassifier(n_estimators=500,bootstrap=True)
clf3 = GaussianNB()
clf4 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,loss="exponential")
clf5 = SGDClassifier(loss="log", alpha=0.01, n_iter=500, fit_intercept=True)
clf6 = KNeighborsClassifier(algorithm="auto",n_jobs=-1,n_neighbors=2,weights="distance")

##Setting up joint classifier
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),('gbc',clf4),('sgd',clf5),('knn',clf6)], voting='soft',weights=[2,3,1,5,0,0])#weights represent how much weight to assign to each model type

for clf, label in zip([clf1, clf2, clf3, clf4,clf5,clf6,eclf], ['Logistic Regression', 'Random Forest', 'Gaussian Naive Bayes', 'GradientBoosting','Stochastic Gradient','KNN','Ensemble']):
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='roc_auc') ##used ROC_AUC as the metric to maximize during crossvalidation
    print("ROC_AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

##fitting the combined classifier
eclf = eclf.fit(X,y)
y_predicted_train= eclf.predict_proba(X)#predicted probability for training set

##going through thresholding to obtain the best threshold probability based off the training set
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


y_predicted_test= eclf.predict_proba(X_test)#predicted probability for test set
predicted_y_test=np.array([1 if x > final_threshold else 0 for x in list(y_predicted_test[:,1])])#using threshold obtained from training set, classify your predictions

#disclose relevant statistics
cm =ConfusionMatrix(y_test, predicted_y_test)
cm.print_stats()
acc=accuracy_score(y_test, predicted_y_test) 
cmatrix=confusion_matrix(y_test, predicted_y_test)
ROI=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)

