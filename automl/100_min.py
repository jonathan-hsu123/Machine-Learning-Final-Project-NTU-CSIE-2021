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
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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
# print(X_train.head(20))
# model = LGBMClassifier(n_estimators=100, random_state=123)
# model = AdaBoostClassifier(n_estimators=200, random_state=42)
# model = RandomForestClassifier(n_estimators=200, random_state=123)
model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=6000, memory_limit=None)
model.fit(X_train, y_train)
y_pre = model.predict(X_test)
y_pre[y_pre == 'No Churn'] = 0
y_pre[y_pre == 'Competitor'] = 1
y_pre[y_pre == 'Dissatisfaction'] = 2
y_pre[y_pre == 'Attitude'] = 3
y_pre[y_pre == 'Price'] = 4
y_pre[y_pre == 'Other'] = 5
test_id['Churn Category'] = y_pre
print(model.leaderboard())
print(model.show_models())
model.cv_results_
model.performance_over_time_.plot(
    x='Timestamp',
    kind='line',
    legend=True,
    title='Auto-sklearn accuracy over time',
    grid=True,
)
plt.savefig("100_min_pic")
test_id.to_csv("100min.csv", index=False)