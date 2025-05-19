#Import all the necessary libraries
import warnings
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import autosklearn.regression
import sklearn.preprocessing
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
import autosklearn.regression
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.model_selection import train_test_split, StratifiedKFold

#Load resistance data
resistance_data = pd.read_excel('MIC_Data_Suzuki2014.xlsx', index_col=0) # normalized IC50 values

# Delete the columns (strains) 'CL1' to 'CL4' as they aren't a part of the expression data.
resistance_data.drop(['CL1','CL2','CL3','CL4'], inplace=True)
resistance_data

# load expression data
expression = pd.read_excel('Suzuki_expression.xlsx', index_col=0, skiprows=1)

# Delete the column (strain) 'Parent (2nd)' as it isn't part of the resistance data.
del expression['Parent (2nd)']

expression

# Create new expression data containing the 8 genes used for regression in the Suzuki et al. paper.
## data was fit using the following eight genes: acrB, ompF, cyoC, pps, tsx, oppA, folA and pntB
eight_genes = ['acrB', 'ompF', 'cyoC', 'pps', 'tsx', 'oppA', 'folA', 'pntB']

expression_red = expression.loc[eight_genes]
expression_red


#Transpose the expression matrix
X = expression_red.T.iloc[:, :]

# Create y matrix
y = resistance_data.reindex(X.index)

# Create y_ENX matrix to see fitting results for one drug type

y_ENX = y['ENX'] #can change the antibiotic name according to each run


X_train, X_test, y_train, y_test = train_test_split(X, y_NM, test_size = 0.2, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


reg = autosklearn.regression.AutoSklearnRegressor(
                            time_left_for_this_task=18000,
                            max_models_on_disc=5,
                            memory_limit = 10240,
                            ensemble_size = 3,
                            metric = r2,
                            scoring_functions=[r2]
                            )

reg.fit(X=X_train, y=y_train, X_test=X_test, y_test=y_test)

#get model stats and export result and leaderboard in an excel file
reg.sprint_statistics()

df_cv_results = pd.DataFrame(reg.cv_results_).sort_values(by = 'mean_test_score', ascending = False)
df_cv_results

df_cv_results.to_excel("result_regression_ENX.xlsx")

reg.leaderboard(detailed = True, ensemble_only=False)

df_cv_leaderboard = pd.DataFrame(reg.leaderboard(detailed = True, ensemble_only=False))
df_cv_leaderboard

df_cv_leaderboard.to_excel("leaderboard_regression_ENX.xlsx")

reg.get_models_with_weights()

#Save the model
joblib.dump(reg,'regression_model_ENX.joblib')

#load the model
reg2 = joblib.load('regression_model_ENX.joblib')

#load model stats and parameters
reg2.sprint_statistics()
reg2.get_params


# Predict using the model for test, train and entire dataset.
y_train_pred = reg2.predict(X_train)
y_test_pred = reg2.predict(X_test)
y_pred = reg2.predict(X)

# Calculate metrics for training set
r2_train = r2_score(y_train, y_train_pred)
print("Training Set Metrics:")
print("R-squared:", r2_train)

# Calculate metrics for test set
r2_test = r2_score(y_test, y_test_pred)
print("\nTest Set Metrics:")
print("R-squared:", r2_test)

# Calculate metrics for the entire dataset
r2 = r2_score(y_NM, y_pred)
print("Metrics for the Entire Dataset:")
print("R-squared:", r2)

