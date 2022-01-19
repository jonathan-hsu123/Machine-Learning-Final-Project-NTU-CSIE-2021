import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
demo = pd.read_csv('demographics.csv')
loca = pd.read_csv('location.csv')
sat = pd.read_csv('satisfaction.csv')
service = pd.read_csv('services.csv')
sta = pd.read_csv('status.csv')
df = pd.merge(demo, sat, on="Customer ID", how="outer")
df = pd.merge(df, loca, on="Customer ID", how="outer")
df = pd.merge(df, service, on="Customer ID", how="outer")
pop = pd.read_csv('population.csv')
df = pd.merge(df, pop, on="Zip Code", how="outer")
df = pd.merge(df, sta, on="Customer ID", how="outer")

df = df.drop(columns=['ID', 'Count_x', 'Count_y', 'Count', 'Country', 'State', 'Lat Long'])

# rule based part
df.loc[(df['Age'] < 30) & (df['Under 30'].isna()), 'Under 30'] = 'Yes'
df.loc[(df['Age'] >= 30) & (df['Under 30'].isna()), 'Under 30'] = 'No'
df.loc[(df['Age'] >= 65) & (df['Senior Citizen'].isna()), 'Senior Citizen'] = 'Yes'
df.loc[(df['Age'] < 65) & (df['Senior Citizen'].isna()), 'Senior Citizen'] = 'No'
df.loc[(df['Number of Dependents'] > 0) & (df['Dependents'].isna()), 'Dependents'] = 'Yes'
df.loc[(df['Number of Dependents'] == 0) & (df['Dependents'].isna()), 'Dependents'] = 'No'
df.loc[(df['Dependents'] == 'No') & (df['Number of Dependents'].isna()), 'Number of Dependents'] = 0.0

# add average part
df['Monthly Refunds'] = df['Total Refunds'] / df['Tenure in Months']
df['Monthly Extra Data Charges'] = df['Total Extra Data Charges'] / df['Tenure in Months']
df['Monthly Long Distance Charges'] = df['Total Long Distance Charges'] / df['Tenure in Months']

#encoding
df.loc[df['Gender'] == 'Male', 'Gender'] = 0.
df.loc[df['Gender'] == 'Female', 'Gender'] = 1.
df['Gender'] = df['Gender'].astype(float)
yes_no_feature = ['Under 30', 'Senior Citizen', 'Married', 'Dependents', 
'Referred a Friend', 'Phone Service','Multiple Lines', 'Internet Service',
'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support',
'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Paperless Billing'
]
for feature in yes_no_feature:
    df.loc[df[feature] == 'Yes', feature] = 0.
    df.loc[df[feature] == 'No', feature] = 1.
    df[feature] = df[feature].astype(float)
catagorical_feature = ['City', 'Quarter', 'Offer', 'Internet Type', 'Contract', 'Payment Method']
le = LabelEncoder()
for feature in catagorical_feature:
    df[feature] = le.fit_transform(df[feature])
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
Test = pd.merge(df, Test_id, on="Customer ID", how="inner")
Test = Test.drop(columns=['Churn Category'])

Train.to_csv("train.csv", index=False)
Test.to_csv("test.csv", index=False)
Train_label.to_csv("label.csv", index=False)