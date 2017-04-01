import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.svm import SVC
import warnings
import sklearn
import csv
import numpy as np
from sklearn.metrics import accuracy_score
from pandas_confusion import ConfusionMatrix
from sklearn import linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
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
# x_train = predictors
# y_train = target



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



# Gradient Boosting
# build a classifier
clf = GradientBoostingClassifier()
x_train = predictors
y_train = target

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, 5, 10, 20, 50],
              "n_estimators": [20,  50, 100, 200, 300],
              "loss": ["exponential", "deviance"]}

# run randomized search
n_iter_search = 3
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring='roc_auc')

start = time()
random_search.fit(x_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)