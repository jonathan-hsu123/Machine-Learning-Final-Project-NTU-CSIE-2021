from shutil import SameFileError
import numpy as np
from numpy.random import RandomState
import openml
import autosklearn
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from pandas.core.frame import DataFrame
import autosklearn.classification
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
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
mfim = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
meanim = SimpleImputer(missing_values=np.NaN, strategy='mean')
knn = KNNImputer(missing_values=np.NaN)
columns = X_train.columns
for feature in columns:
    if X_train[feature].dtype == float:
        X_train[feature] = meanim.fit_transform(X_train[[feature]]).ravel()
        X_test[feature] = meanim.fit_transform(X_test[[feature]]).ravel()
    else:
        X_train[feature] = knn.fit_transform(X_train[[feature]]).ravel()
        X_test[feature] = knn.fit_transform(X_test[[feature]]).ravel()
model = LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
y_pre[y_pre == 'No Churn'] = 0
y_pre[y_pre == 'Competitor'] = 1
y_pre[y_pre == 'Dissatisfaction'] = 2
y_pre[y_pre == 'Attitude'] = 3
y_pre[y_pre == 'Price'] = 4
y_pre[y_pre == 'Other'] = 5
test_id['Churn Category'] = y_pre
test_id.to_csv("mean_kNN_LGBM.csv", index=False)
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
test_id.to_csv("mean_kNN_ADA.csv", index=False)
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
test_id.to_csv("mean_kNN_tree.csv", index=False)
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
y_pre[y_pre == 'No Churn'] = 0
y_pre[y_pre == 'Competitor'] = 1
y_pre[y_pre == 'Dissatisfaction'] = 2
y_pre[y_pre == 'Attitude'] = 3
y_pre[y_pre == 'Price'] = 4
y_pre[y_pre == 'Other'] = 5
test_id['Churn Category'] = y_pre
test_id.to_csv("mean_kNN_XGB.csv", index=False)