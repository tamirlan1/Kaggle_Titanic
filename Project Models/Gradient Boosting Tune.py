# -*- coding: utf-8 -*-
# modified from the framework that initially appeared here http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html#example-model-selection-randomized-search-py
#we adopted syntax to search for optimal hyperparameters for our gradient boosting classifier
import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier

#obtain data
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
clf = GradientBoostingClassifier()


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
param_dist = {"max_depth": [3, 15, 20, 50],
              "n_estimators": [20, 50, 100, 200, 300],
              "loss": ["exponential", "deviance"]}

# run randomized search
n_iter_search = 3
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring='roc_auc')

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

