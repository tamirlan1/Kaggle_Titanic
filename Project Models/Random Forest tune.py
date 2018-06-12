import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import random
import csv



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

# build a classifier
clf = RandomForestClassifier()


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
param_dist = {"max_depth": [50, None],
              "max_features": sp_randint(15, 50),
              "min_samples_split": sp_randint(100, 1000),
              "min_samples_leaf": sp_randint(100, 1000),
              "bootstrap": [True, False],
              "n_estimators":sp_randint(10, 1000),
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring='roc_auc')

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

# use a full grid over all parameters

# run grid search

param_grid = {"bootstrap": [True, False],
              "n_estimators":[10,30,50,100,200,500]}

grid_search = GridSearchCV(clf, param_grid=param_grid,scoring='roc_auc')
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)