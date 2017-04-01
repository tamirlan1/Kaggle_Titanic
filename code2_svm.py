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
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

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

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
data_test_clean = scaler.fit_transform(data_test_clean)

# Support Vector Machine
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
kernel_range = ['linear', 'rbf']
best_C = 0#10000
best_gamma = 0#0.001
best_score = 0
best_kernel = ''#'rbf'

for kernel in kernel_range:
	print 'KERNEL: ', kernel
	for C in C_range:
		print 'C: ', C
		for gamma in gamma_range:
			print 'gamma: ', gamma
			model_svm = SVC(kernel=kernel, C=C, gamma=gamma)
			model_svm = model_svm.fit(x_train, y_train)
			score = cross_val_score(model_svm, x_train, y_train, cv=5).mean()
			if score > best_score:
				best_score = score
				print 'Best score is: ', best_score
				best_gamma = gamma
				best_C = C
				best_kernel = kernel

print 'Best C: ', best_C
print 'Best Gamma: ', best_gamma
print 'Best Score: ', best_score

model_svm = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma)
model_svm = model_svm.fit(x_train, y_train)
svm_predicted = model_svm.predict(x_train)
print 'Train accuracy: ', accuracy_score(y_train, svm_predicted)

y_predicted_test_final = model_svm.predict(data_test_clean)
data_test_id = data_test['PassengerId'].values
prediction_file = open("RESULTS_SVM_tuned.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
print 'This needs to be 418: ', len(data_test_id)
print 'This needs to be 418: ', len(y_predicted_test_final)
prediction_file_object.writerows(zip(data_test_id, y_predicted_test_final))
prediction_file.close()
#Accuracy = 0.78469

print 'Number of positive predictions is ', sum(y_predicted_test_final), ' out of ', len(y_predicted_test_final), ' predictions'



