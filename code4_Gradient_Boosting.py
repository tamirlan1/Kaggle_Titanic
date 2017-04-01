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

x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=.2, random_state=2016)
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
GB_model = GradientBoostingClassifier(n_estimators=50, loss='deviance', max_depth=3)
    #n_estimators=100, learning_rate=0.1,loss="exponential")
GB_model = GB_model.fit(x_train,y_train)##fit model

y_probability_train = GB_model.predict_proba(x_train)
y_probability_test = GB_model.predict_proba(x_test)

best_threshold = 0.5
best_accuracy = 0
# Find best threshold
for i in xrange(1,100):
    thr = i*0.01
    y_predicted_test = np.array([1 if value > thr else 0 for value in list(y_probability_test[:,1])])
    accuracy = accuracy_score(y_test, y_predicted_test)
    print 'Accuracy: ', accuracy
    # score_tr = rf_model.score(y_test, y_predicted_test)
    # print 'train score: ', score_tr
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = thr

print 'Best threshold: ', best_threshold # 0.73
print 'Best Accuracy: ', best_accuracy # 0.854748603352

y_probability_test_final = GB_model.predict_proba(data_test_clean)
y_predicted_test_final = np.array([1 if value > best_threshold else 0 for value in list(y_probability_test_final[:,1])])

print y_predicted_test_final
data_test_id = data_test['PassengerId'].values
prediction_file = open("RESULTS_Gradient_Boosting_tune.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
print 'This needs to be 418: ', len(data_test_id)
print 'This needs to be 418: ', len(y_predicted_test_final)
prediction_file_object.writerows(zip(data_test_id, y_predicted_test_final))
prediction_file.close()
#Accuracy = 0.79904




print sum(y_predicted_test_final)
# print sum(logr_predictions_test2)
# print sum(logr_predictions_test3)
# print sum(logr_predictions_test4)
