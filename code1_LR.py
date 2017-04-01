import pandas as pd
import statsmodels.formula.api as smf
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.svm import SVC
import warnings
import sklearn
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from pandas_confusion import ConfusionMatrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn import linear_model

warnings.filterwarnings('ignore')


# DATA CLEANING
data0 = pd.read_csv('train.csv')
# Clean unnecessary columns and drop rows with Nan
data_clean = data0.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)

# All the ages with no data -> make the mean of all Ages
mean_age = data_clean['Age'].dropna().mean()
data_clean.iloc[:, 3][data_clean.iloc[:, 3].isnull()] = mean_age

# Find missing values
for i in range(data_clean.shape[1]):
	if sum(data_clean.iloc[:, i].isnull()) > 0:
		print 'For row ', i, 'with column name ', data_clean.columns.values[i], ' number of missing values is: ', sum(data_clean.iloc[:, i].isnull())

data_clean['Age'] = data_clean['Age'].astype('int64')
data_clean['Fare'] = data_clean['Fare'].astype('int64')

lb = LabelEncoder()
data_clean['Sex'] = lb.fit_transform(data_clean['Sex'])
lb2 = LabelEncoder()
data_clean['Embarked'] = lb2.fit_transform(data_clean['Embarked'])

predictors = data_clean.drop('Survived', 1)
target = data_clean.Survived

# x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=.2, random_state=2016)
x_train = predictors
y_train = target



# TEST SET
data_test = pd.read_csv('test.csv')

# Clean unnecessary columns and drop rows with Nan
data_test_clean = data_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)

# Find missing values
# for i in range(data_test_clean.shape[1]):
# 	if sum(data_test_clean.iloc[:, i].isnull()) > 0:
# 		print 'For row ', i, 'with column name ', data_test_clean.columns.values[i], ' number of missing values is: ', sum(data_test_clean.iloc[:, i].isnull())

# print 'Age:', data_test_clean.iloc[:, 2]
# print 'Fare:', data_test_clean.iloc[:, 5]

# All the ages with no data -> make the mean of all Ages
mean_age = data_test_clean['Age'].dropna().mean()
data_test_clean.iloc[:, 2][data_test_clean.iloc[:, 2].isnull()] = mean_age

mean_fare = data_test_clean['Fare'].dropna().mean()
data_test_clean.iloc[:, 5][data_test_clean.iloc[:, 5].isnull()] = mean_fare

# data_test_clean = data_test_clean.dropna()
data_test_clean['Age'] = data_test_clean['Age'].astype('int64')
data_test_clean['Fare'] = data_test_clean['Fare'].astype('int64')
data_test_clean['Sex'] = lb.fit_transform(data_test_clean['Sex'])
data_test_clean['Embarked'] = lb2.fit_transform(data_test_clean['Embarked'])



# LOGISTIC REGRESSION
logr = LogisticRegression(class_weight ='balanced')
logr = logr.fit( x_train, y_train )
logr_predictions_train = logr.predict(x_train)
accuracy_train = accuracy_score(y_train, logr_predictions_train)
print accuracy_train #0.78675
cm1 = ConfusionMatrix(y_train, logr_predictions_train)
# cm1.print_stats()


logr_predictions_test = logr.predict(data_test_clean)
data_test_id = data_test['PassengerId'].values

prediction_file = open("RESULTS_LR4.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
print 'This needs to be 418: ', len(data_test_id)
print 'This needs to be 418: ', len(logr_predictions_test)
prediction_file_object.writerows(zip(data_test_id, logr_predictions_test))
prediction_file.close()
#Accuracy = 0.73206


# REGULARIZATION - search for the best parameter
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid)
GridSearchCV(cv=None,
       estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
          penalty='l2', tol=0.0001),
       param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
clf = clf.fit(x_train, y_train)
logr_predictions_train2 = clf.predict(x_train)
logr_predictions_test2 = clf.predict(data_test_clean)
# cm2 =ConfusionMatrix(y_test, logr_predictions_train2)
# cm2.print_stats()
accuracy_train2 = accuracy_score(y_train, logr_predictions_train2) 
print 'accuracy2: ', accuracy_train2 #0.8013

prediction_file = open("RESULTS_LR_regul.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
print 'This needs to be 418: ', len(data_test_id)
print 'This needs to be 418: ', len(logr_predictions_test2)
prediction_file_object.writerows(zip(data_test_id, logr_predictions_test2))
prediction_file.close()
#Accuracy = 0.74641


# Cross validation
logisticCV = linear_model.LogisticRegressionCV(class_weight ='balanced',scoring='roc_auc')
logisticCV = logisticCV.fit(x_train, y_train)
logr_predictions_train3 = logisticCV.predict(x_train)
logr_predictions_test3 = logisticCV.predict(data_test_clean)
accuracy_train3 = accuracy_score(y_train, logr_predictions_train3) 
print 'accuracy3: ', accuracy_train3 # 0.789001122334

prediction_file = open("RESULTS_LR_CV.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
print 'This needs to be 418: ', len(data_test_id)
print 'This needs to be 418: ', len(logr_predictions_test3)
prediction_file_object.writerows(zip(data_test_id, logr_predictions_test3))
prediction_file.close()
#Accuracy = 0.73206


# CHOOSE BEST THRESHOLD
logistic = linear_model.LogisticRegression(class_weight ='balanced')
logistic = logistic.fit(x_train,y_train)
y_predicted_train = logistic.predict_proba(x_train)#predicted class for training set

#obtain critical threshold
best_accuracy=0
final_threshold =0.5
for x in xrange(1,100):
    thresh = 0.01*x
    logr_predictions_train4=np.array([1 if x > thresh else 0 for x in list(y_predicted_train[:,1])])
    acc = accuracy_score(y_train, logr_predictions_train4)
    if acc>best_accuracy:
        best_accuracy=acc
        final_threshold=thresh
print 'best accuracy: ', best_accuracy #0.8170594837
print 'threshold: ', final_threshold
##get predictions on test set and use the optimal threshold determined on the training set to determine classes

y_predicted_test= logistic.predict_proba(data_test_clean)#predicted class for test set
logr_predictions_test4=np.array([1 if x > final_threshold else 0 for x in list(y_predicted_test[:,1])])

##obtain relevant statistics

prediction_file = open("RESULTS_LR_tuned_thr.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
print 'This needs to be 418: ', len(data_test_id)
print 'This needs to be 418: ', len(logr_predictions_test4)
prediction_file_object.writerows(zip(data_test_id, logr_predictions_test4))
prediction_file.close()
# Accuracy = 0.77990

# print sum(logr_predictions_test)
# print sum(logr_predictions_test2)
# print sum(logr_predictions_test3)
# print sum(logr_predictions_test4)
