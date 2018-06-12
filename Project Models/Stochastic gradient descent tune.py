import numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
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
from sklearn.linear_model import SGDClassifier

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

#We manually tried various loss functions here but the model was suboptimal by all accounts and varying malleable hyperparameters did not seem to change much
clf = SGDClassifier(loss="perceptron", alpha=0.01, n_iter=500, fit_intercept=True)
clf = clf.fit(X,y)

y_predicted = clf.predict(X_test)#predicted class
cm =ConfusionMatrix(y_test, y_predicted)
cm.print_stats()
acc=accuracy_score(y_test, y_predicted) 
#print(classification_report(y_test, y_predicted))
cmatrix=confusion_matrix(y_test, y_predicted)
ROI=cmatrix[1,1]*100 + cmatrix[0,0]*15 + cmatrix[0,1]*(-15) + cmatrix[1,0]*(-30)

