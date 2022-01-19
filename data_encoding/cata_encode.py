from shutil import SameFileError
from tkinter import N
import numpy as np
from numpy.random import RandomState
import openml
import autosklearn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from pandas.core.frame import DataFrame
import autosklearn.classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
label = pd.read_csv("label.csv")
train_id = train.iloc[:, 0].to_frame()
test_id = test.iloc[:, 0].to_frame()
train = train.iloc[:, 1:]
test = test.iloc[:, 1:]
X_train = train
y_train = label.iloc[:, -1]
X_test = test
le = LabelEncoder()
for col in X_train.columns:
    if(X_train[col].dtypes == object):
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.fit_transform(X_test[col])
    else:
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_test[col] = X_test[col].fillna(X_test[col].mean())
# print(X_train.head())
model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=50)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
y_pre[y_pre == 'No Churn'] = 0
y_pre[y_pre == 'Competitor'] = 1
y_pre[y_pre == 'Dissatisfaction'] = 2
y_pre[y_pre == 'Attitude'] = 3
y_pre[y_pre == 'Price'] = 4
y_pre[y_pre == 'Other'] = 5
test_id['Churn Category'] = y_pre
test_id.to_csv("cata_LGBM.csv", index=False)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
y_pre[y_pre == 'No Churn'] = 0
y_pre[y_pre == 'Competitor'] = 1
y_pre[y_pre == 'Dissatisfaction'] = 2
y_pre[y_pre == 'Attitude'] = 3
y_pre[y_pre == 'Price'] = 4
y_pre[y_pre == 'Other'] = 5
test_id['Churn Category'] = y_pre
test_id.to_csv("cata_ADA.csv", index=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
y_pre[y_pre == 'No Churn'] = 0
y_pre[y_pre == 'Competitor'] = 1
y_pre[y_pre == 'Dissatisfaction'] = 2
y_pre[y_pre == 'Attitude'] = 3
y_pre[y_pre == 'Price'] = 4
y_pre[y_pre == 'Other'] = 5
test_id['Churn Category'] = y_pre
test_id.to_csv("cata_tree.csv", index=False)
# model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=3600, memory_limit=None)
