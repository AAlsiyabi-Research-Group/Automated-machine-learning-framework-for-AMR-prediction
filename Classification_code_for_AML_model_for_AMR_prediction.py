#Import all the necessary libraries
import autosklearn
import joblib
import pandas as pd
import numpy as np
import warnings

from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import autosklearn.classification
import autosklearn.classification as classifier
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import (accuracy,
                                 f1,
                                 roc_auc,
                                 precision,
                                 average_precision,
                                 recall,
                                 log_loss,
                                 r2,
                                 mean_squared_error,
                                 mean_absolute_error,
                                 )
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.model_selection import train_test_split, StratifiedKFold

#Load resistance data
resistance_data = pd.read_excel('resistance_data.xlsx', index_col=0) # normalized IC50 values

# Delete the columns (strains) 'CL1' to 'CL4' as they aren't a part of the expression data.
resistance_data.drop(['CL1','CL2','CL3','CL4'], inplace=True)
resistance_data

# load expression data
expression = pd.read_excel('expression_data.xlsx', index_col=0, skiprows=1)

# Delete the column (strain) as it isn't part of the resistance data.
del expression['Parent (2nd)']

expression

# Create new expression data containing the 8 genes used for regression in the Suzuki et al. paper.
## data was fit using the following eight genes: acrB, ompF, cyoC, pps, tsx, oppA, folA and pntB
eight_genes = ['acrB', 'ompF', 'cyoC', 'pps', 'tsx', 'oppA', 'folA', 'pntB']

expression_red = expression.loc[eight_genes]

#Transpose the expression matrix
X = expression_red.T.iloc[:, :]

# Create y matrix
y = resistance_data.reindex(X.index)

# Create y_ENX matrix to see fitting results for one drug type

y_ENX = y['ENX']

# Create histogram to check distribution of MICs.
hist, bins = np.histogram(y_ENX, bins=8)

print("Histogram:", hist)
print("Bins:", bins)

# prepare y for classification
y_new = y_ENX
y_new[y_new <= 1] = 0
y_new[y_new >= 2] = 1
y_new

X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size = 0.2, random_state=1, stratify=y_new)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

skf = StratifiedKFold(n_splits=3)

clf = AutoSklearnClassifier(time_left_for_this_task=18000,
                            memory_limit = 10240, #memory limit can also be set up as None if the 10240 does not work.
                            resampling_strategy=skf,
                            ensemble_kwargs={'ensemble_size': 3},
                            metric=roc_auc,
                            scoring_functions=[roc_auc, average_precision, accuracy, f1, precision, recall, log_loss])

clf.fit(X=X_train, y=y_train, X_test=X_test, y_test=y_test)

#get model stats and export result and leaderboard in an excel file
clf.sprint_statistics()

df_cv_results = pd.DataFrame(clf.cv_results_).sort_values(by = 'mean_test_score', ascending = False)
df_cv_results

df_cv_results.to_excel("result_ENX.xlsx")

clf.leaderboard(detailed = True, ensemble_only=False)

df_cv_leaderboard = pd.DataFrame(clf.leaderboard(detailed = True, ensemble_only=False))
df_cv_leaderboard

df_cv_leaderboard.to_excel("leaderboard_ENX.xlsx")

clf.get_models_with_weights()

#Save the model
joblib.dump(clf,'model_ENX.joblib')

#load the model
clf2 = joblib.load('model_ENX.joblib')

#load model stats and parameters
clf2.sprint_statistics()

clf2.get_params

#refit model for prediction
clf2.refit(X=X_train, y=y_train)

clf2.predict(X)

import sklearn
y_hat_test = clf2.predict(X_test)
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat_test))

y_hat_train = clf2.predict(X_train)
print("Accuracy score", sklearn.metrics.accuracy_score(y_train, y_hat_train))

# Assuming y_all and y_hat_all are the combined true and predicted labels across all datasets
f1_overall = f1_score(y_new, y_hat_full)
print("Overall F1 Score:", f1_overall)

from sklearn.metrics import f1_score

# Calculate F1 score for the test set
f1_test = f1_score(y_test, y_hat_test)
print("F1 Score for the test set:", f1_test)

# Calculate F1 score for the training set
f1_train = f1_score(y_train, y_hat_train)
print("F1 Score for the training set:", f1_train)
