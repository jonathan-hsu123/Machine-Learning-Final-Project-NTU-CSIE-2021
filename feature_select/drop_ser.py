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
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

demo = pd.read_csv('demographics.csv')
loca = pd.read_csv('location.csv')
sat = pd.read_csv('satisfaction.csv')
service = pd.read_csv('services.csv')
sta = pd.read_csv('status.csv')
df = pd.merge(demo, sat, on="Customer ID", how="outer")
df = pd.merge(df, loca, on="Customer ID", how="outer")
# df = pd.merge(df, service, on="Customer ID", how="outer")
pop = pd.read_csv('population.csv')
df = pd.merge(df, pop, on="Zip Code", how="outer")
df = pd.merge(df, sta, on="Customer ID", how="outer")

df = df.drop(columns=['ID', 'Count_x', 'Count_y', 'Country', 'State', 'Lat Long'])

# rule based part
df.loc[(df['Age'] < 30) & (df['Under 30'].isna()), 'Under 30'] = 'Yes'
df.loc[(df['Age'] >= 30) & (df['Under 30'].isna()), 'Under 30'] = 'No'
df.loc[(df['Age'] >= 65) & (df['Senior Citizen'].isna()), 'Senior Citizen'] = 'Yes'
df.loc[(df['Age'] < 65) & (df['Senior Citizen'].isna()), 'Senior Citizen'] = 'No'
df.loc[(df['Number of Dependents'] > 0) & (df['Dependents'].isna()), 'Dependents'] = 'Yes'
df.loc[(df['Number of Dependents'] == 0) & (df['Dependents'].isna()), 'Dependents'] = 'No'
df.loc[(df['Dependents'] == 'No') & (df['Number of Dependents'].isna()), 'Number of Dependents'] = 0.0

#encoding
# df.loc[df['Gender'] == 'Male', 'Gender'] = 0.
# df.loc[df['Gender'] == 'Female', 'Gender'] = 1.
# df['Gender'] = df['Gender'].astype(float)
# yes_no_feature = ['Under 30', 'Senior Citizen', 'Married', 'Dependents'
# ]
# for feature in yes_no_feature:
#     df.loc[df[feature] == 'Yes', feature] = 0.
#     df.loc[df[feature] == 'No', feature] = 1.
#     df[feature] = df[feature].astype(float)
# catagorical_feature = ['City']
le = LabelEncoder()
# for feature in catagorical_feature:
#     df[feature] = le.fit_transform(df[feature])
# print(df.dtypes)

#use satisfaction score to fill nan churn

Train_id = pd.read_csv('Train_IDs.csv')
Train = pd.merge(df, Train_id, on="Customer ID", how="inner")

# Train.loc[(Train['Churn Category'].isna()) & (Train['Satisfaction Score'] < 3), 'Churn Category'] = 'Dissatisfaction'
# Train.loc[(Train['Churn Category'].isna()) & (Train['Satisfaction Score'] == 3), 'Churn Category'] = 'Competitor'
# Train.loc[(Train['Churn Category'].isna()) & (Train['Satisfaction Score'] > 4), 'Churn Category'] = 'No Churn'

Train = Train[~Train['Churn Category'].isna()]
Train_label = Train[["Customer ID", 'Churn Category']]
Train = Train.drop(columns=['Churn Category'])
Test_id = pd.read_csv('Test_IDs.csv')
Test = pd.merge(df, Test_id, on="Customer ID", how="right")
Test = Test.drop(columns=['Churn Category'])

train_id = Train.iloc[:, 0].to_frame()
test_id = Test.iloc[:, 0].to_frame()
train = Train.iloc[:, 1:]
test = Test.iloc[:, 1:]
X_train = train
y_train = Train_label.iloc[:, -1]
X_test = test
for col in X_train.columns:
    if(X_train[col].dtypes == object):
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.fit_transform(X_test[col])
    else:
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_test[col] = X_test[col].fillna(X_test[col].mean())
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
test_id.to_csv("ser_LGBM.csv", index=False)
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
test_id.to_csv("ser_ADA.csv", index=False)
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
test_id.to_csv("ser_tree.csv", index=False)