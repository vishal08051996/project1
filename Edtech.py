#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:30:25 2022

@author: venky
"""
## Load the data set using MySQL servier

import mysql.connector
#establishing the connection
conn = mysql.connector.connect(
   user='root', password='Venky@1985', host='localhost', database='student')


#Creating a cursor object using the cursor() method
cursor = conn.cursor()

#Retrieving single row
sql = '''SELECT * from MOCK_DATA'''

#Executing the query
cursor.execute(sql)

#Fetching 1st row from the table
result = cursor.fetchall();
print(result)

"""#Closing the connection
conn.close()"""
import pandas as pd ## Data manipulation
data = pd.DataFrame(result, columns = ['s.no', 'Institute', 'Subject', 'Location', 'Trainer_Qualification',
       'Online_classes', 'Offline_classes', 'Trainer_experiance',
       'Course_level', 'Course_hours', 'Course_rating', 'Rental_permises',
       'Trainer_slary', 'Maintaince_cost', 'Non_teaching_staff_salary',
       'Placements', 'Certificate', 'Price'])



## or use my computor to load the data set.
import pandas as pd
data = pd.read_csv("C:\\Users\\venki\\OneDrive\\Desktop\\Datascience360\\VS-Deploy-11-07\\MOCK_DATA.csv")

data['Location'].value_counts()

import numpy as np ## numerical(mathematical) calculations

## Understanding the data
data.info()  ## information about the null,data type, memory
x = data.describe() ## statistical information
data.shape ## (525, 18)
data.columns
"""['s.no', 'Institute', 'Subject', 'Location', 'Trainer_Qualification',
       'Online_classes', 'Offline_classes', 'Trainer_experiance',
       'Course_level', 'Course_hours', 'Course_rating', 'Rental_permises',
       'Trainer_slary', 'Maintaince_cost', 'Non_teaching_staff_salary',
       'Placements', 'Certificate', 'Price']"""

data.drop(['s.no'], axis = 1, inplace = True)
data.shape # (525,17)

## Data cleaning

data.duplicated().sum() ## no duplicates
data.isna().sum() # no null values

## Label encoder (Converting categorical into numeric)
cols = ['Institute', 'Subject', 'Location', 'Trainer_Qualification','Online_classes', 'Offline_classes',
         'Course_level','Placements', 'Certificate']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Instantiate the encoders
encoders = {column: le for column in cols}

for column in cols:
    data[column] = encoders[column].fit_transform(data[column])

data.info()

## check ing outliers
import seaborn as sns
import matplotlib.pyplot as plt

for i in cols:
    sns.boxplot(data[i]); plt.show() 

bx = sns.boxplot(data = data, orient ="h", palette = "Set2")
plt.xticks(rotation = 45)
data.var()

sns.pairplot(data, hue="class", diag_kind="hist")

sns.jointplot(x = 'Price', y ='Course_hours', data = data)

sns.heatmap(data.corr(), annot=True)


## Certificate column has zero variance, so drop it
data.drop(['Certificate'], axis = 1, inplace = True)

import statsmodels.formula.api as smf 
         
ml1 = smf.ols('Price ~ Institute + Subject + Location + Trainer_Qualification + Online_classes + Offline_classes + Course_level + Course_hours + Course_rating + Rental_permises + Trainer_slary + Maintaince_cost + Non_teaching_staff_salary + Placements', data = data).fit() # regression model

# Summary
ml1.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = data[list(data.select_dtypes(include = ['int64', 'float64']).columns)]

# Profit feature is dependent or out put feature so we are deleting
X = X.drop('Price', axis = 1)

## VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

## calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)

# Split the data set into train(80% of the data) and test(20% of the data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.drop("Price", axis = 1), data.Price, test_size = 0.2, random_state = 42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
rmse =lambda y, y_hat: np.sqrt(mean_squared_error(y, y_hat))

lm = LinearRegression()
lm.fit(x_train, y_train)


y_pred_test = lm.predict(x_test)

result_test = pd.DataFrame({'Actual':y_test, "Predicted": y_pred_test})
result_test.head(10)

## importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# predicting the accuracy score
score_test = r2_score(y_test, y_pred_test)

print('R2 score(test): ', score_test)
print('Mean squared error(test): ', mean_squared_error(y_test, y_pred_test))
print('Root Mean squared error(test): ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

""" 
R2 score(test):  0.9758756217887679
Mean squared error(test):  2720.581687776762
Root Mean squared error(test):  52.1591956204921
"""

y_pred_train = lm.predict(x_train)

result_train = pd.DataFrame({'Actual':y_train, "Predicted": y_pred_train})
result_train.head(10)

score_train = r2_score(y_train, y_pred_train)

print('R2 score(train): ', score_train)
print('Mean squared error(train): ', mean_squared_error(y_train, y_pred_train))
print('Root Mean squared error(train): ', np.sqrt(mean_squared_error(y_train, y_pred_train)))

"""
R2 score(train):  0.9916066817263751
Mean squared error(train):  1000.358360676545
Root Mean squared error(train):  31.62844227394933
"""
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred_test)
plt.ylabel("y_pred_test")
plt.xlabel("y_test")


import pickle

pickle.dump(lm, 'data_model.pkl')

pickle.dump(lm, open('data_model.pkl', 'wb'))

# Load the model from disk
model = pickle.load(open("data_model.pkl", "rb"))

result = model.score(x_test, y_test)
print(result)

#Model predict the course price correctly based on features(15).

import pickle
data.to_csv("data.csv", index=False)
pickle.dump(lm, open('data.pkl','wb'))


