# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:01:56 2021

@author: Anuj
"""
#Importing necessary library
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_csv("diabetes.csv")
print(df.head())

'''
Clearning null values and zeros from data
Checking is there any null value
Null values existence can affect our prediction accuracy, hence we need to remove null values if it exists.
'''
print(df.isnull().sum())

'''
Checking null isn't sufficient
So no null values were found but that's not enough, as there may be some zeros in columns like Blood Pressure which is not normal and can affect our prediction accuracy.

Zeros
Now let's check is there any zeros in columns like Blood Pressure, Insulin, etc:
'''
for i in df.columns:
    print(i,len(df[df[i] == 0]))

'''
Zeros detected
As zero were detected in columns like Pregnancies, Glucose, BloodPressure, SkinThickness, and Insulin,
we need to remove it. We will do this by replacing zeros into null values then simply we will use ffill or bfill method.
But columns like Pregnancies can have zero so we won't remove zeros from this type of columns.
'''
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0,np.nan)
print(df.isnull().sum())

'''
Removing null values
'''
df = df.fillna(method = 'ffill')
df = df.fillna(method = 'bfill')
print(df.isnull().sum())

'''
Checking outliers
Boxplot
Now let's check outliers in our data by simply using boxplot.
'''
fig1, axes1 =plt.subplots(4,2,figsize=(14, 19))
list1_col=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
for i in range(len(list1_col)):
    row=i//2
    col=i%2
    ax=axes1[row,col]
    sns.boxplot(df[list1_col[i]],ax=ax).set(title='PRODUCT BY ' + list1_col[i].upper())
    
'''
Observation(Outliers):
We can clearly observe that insulin reaching 800 which means there is soo many outliers 
in df.Insulin. Similarly, in SkinThickness columns also, outlier exists. We have to remove them 
for better prediction (better accuracy).

Removing outlier
Now let's identify and remove outlier. We will use quantile method to remove outliers.
'''
print(df.Insulin.shape)
print(df.SkinThickness.shape)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

df = df[~((df.iloc[:,4:5] < (Q1 - 1.5 * IQR)) |(df.iloc[:,4:5] > (Q3 + 1.5 * IQR))).any(axis=1)]
df = df[~((df.iloc[:,3:4] < (Q1 - 1.5 * IQR)) |(df.iloc[:,3:4] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(df.Insulin.shape)
print(df.SkinThickness.shape)

'''
Observing outliers again with Boxplot
Let's observe again outliers using boxplot. We are observing again to make sure that 
all outliers which we want to remove is removed.
'''
fig1, axes1 =plt.subplots(4,2,figsize=(14, 19))
list1_col=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
for i in range(len(list1_col)):
    row=i//2
    col=i%2
    ax=axes1[row,col]
    sns.boxplot(df[list1_col[i]],ax=ax).set(title='PRODUCT BY ' + list1_col[i].upper())
    
'''
Observation:
As we can see that outlier in insulin and skinthickness is removed so we can proceed further and predict.
'''
'''
Prediction
'''
x = df.drop("Outcome", axis=1)
y = df.Outcome

'''
Train data and Test data
Let's split data into train and test:
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 
print(x_train.head())

'''
Logistic Regression
Let's predict using Logistic Regression
'''
lr =LogisticRegression(solver='liblinear')
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
'''
AUC
Now let's check auc
'''
y_pred_proba = lr.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


'''
Decision Tree Classifier
Let's predict using decision tree classifier and check it's accuracy, recall and precision.
'''
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred1 = dt.predict(x_test)

print(classification_report(y_test, y_pred1))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
print("Precision:",metrics.precision_score(y_test, y_pred1))
print("Recall:",metrics.recall_score(y_test, y_pred1))
'''
AUC
Now let's check its AUC
'''
y_pred_proba = dt.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


'''
Random Forest Classifier
Let's predict using random forest classifier and check it's accuracy, recall and precision.
'''
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred2 = rf.predict(x_test)

print(classification_report(y_test, y_pred2))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred2))
print("Precision:",metrics.precision_score(y_test, y_pred2))
print("Recall:",metrics.recall_score(y_test, y_pred2))
'''
AUC
Now let's check its AUC
'''
y_pred_proba = rf.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

'''
Final Result
So as we can see from above three prediction, Logistic regression has more accuracy. 
Also Logistic regression has more auc than others but auc of Logistic Regression 
and auc of Random Forest Classifier are almost similar. While Logistic regression 
and Random forest classifier has almost similar precision 
(Precision of Logistic regression is slightly higher than Random forest classifier's precision) 
 If we talk about Recall, then Decision tree classifier has more recall than others.
'''

























