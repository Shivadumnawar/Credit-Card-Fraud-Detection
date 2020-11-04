# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:58:30 2020

@author: shiva dumnawar
"""

import pandas as pd
import numpy as np
import seaborn as sns                      
import matplotlib.pyplot as plt


df= pd.read_csv('creditcard.csv')

df.head()

df.info()

df.isnull().sum() 
# no null values

des= df.describe()

df['Class'].value_counts()

print('valid transactions :',round(df['Class'].value_counts()[0]/len(df)*100,2),'%')
print('fraud transactions :',round(df['Class'].value_counts()[1]/len(df)*100,2),'%')

# dataset is highly imbalanced

sns.countplot(x='Class', data= df, palette='Set1')

sns.displot(x= 'Amount', data= df)

sns.displot(x= 'Time', data=df)

# scaling Amount and Time

from sklearn.preprocessing import StandardScaler

ss= StandardScaler()

df['Amount']= ss.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time']= ss.fit_transform(df['Time'].values.reshape(-1,1))

df.head(10)

# Using the given imbalanced dataset causes overfitting. 
# To avoid overfitting, new sub-sample is created with 492 'fraud' and 492 'no fraud' transactions.

# shuffle the dataframe
df= df.sample(frac=1)

fraud_df= df.loc[df['Class']==1]
no_fraud_df= df.loc[df['Class']==0][:492]

norm_distributed_df= pd.concat([fraud_df, no_fraud_df])

new_df= norm_distributed_df.sample(frac=1)

sns.countplot(x='Class', data= new_df, palette='Set1')
# data is normally distributed

X= new_df.iloc[:, :-1]
y= new_df.iloc[:, -1].values

#
from sklearn.ensemble import ExtraTreesClassifier
model= ExtraTreesClassifier()
model.fit(X,y)

model.feature_importances_

feat_importances= pd.Series(model.feature_importances_, index= X.columns)

plt.figure(figsize=(12,8))
feat_importances.nlargest(25).plot(kind='barh')

# correlation
c= new_df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(c, cmap= 'coolwarm', annot=False )

# V4, V14, V12, V11, V17, V16 these featues are highly correlated to the class feature

# Negative correlation - V14, V12, V17, V16
# Positive correlation - V4, V11

f, axes= plt.subplots(nrows=2, ncols=3, figsize=(16,12))

sns.boxplot(x= 'Class', y= 'V4', data= new_df, palette= 'Set1', ax= axes[0][0])
axes[0][0].set_title('V4 vs Class correlation')

sns.boxplot(x= 'Class', y= 'V14', data= new_df, palette= 'Set1', ax= axes[0][1])
axes[0][1].set_title('V14 vs Class correlation')

sns.boxplot(x= 'Class', y= 'V12', data= new_df, palette= 'Set1', ax= axes[0][2])
axes[0][2].set_title('V12 vs Class correlation')

sns.boxplot(x= 'Class', y= 'V11', data= new_df, palette= 'Set1', ax= axes[1][0])
axes[1][0].set_title('V11 vs Class correlation')

sns.boxplot(x= 'Class', y= 'V17', data= new_df, palette= 'Set1', ax= axes[1][1])
axes[1][1].set_title('V17 vs Class correlation')

sns.boxplot(x= 'Class', y= 'V16', data= new_df, palette= 'Set1', ax= axes[1][2])
axes[1][2].set_title('V16 vs Class correlation')

# Using Clip() method, outliers are removed 
new_df['V4']= new_df['V4'].clip(lower=new_df['V4'].quantile(.2), upper= new_df['V4'].quantile(.75))
new_df['V14']= new_df['V14'].clip(lower=new_df['V14'].quantile(.2), upper= new_df['V14'].quantile(.75))
new_df['V12']= new_df['V12'].clip(lower=new_df['V12'].quantile(.2), upper= new_df['V12'].quantile(.75))
new_df['V11']= new_df['V11'].clip(lower=new_df['V11'].quantile(.2), upper= new_df['V11'].quantile(.75))
new_df['V17']= new_df['V17'].clip(lower=new_df['V17'].quantile(.2), upper= new_df['V17'].quantile(.75))
new_df['V16']= new_df['V16'].clip(lower=new_df['V16'].quantile(.2), upper= new_df['V16'].quantile(.75))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=15)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# logistic regression
log_reg= LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred= log_reg.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))

print('accuracy score : ' , accuracy_score(y_test, y_pred))

# KNN

clf= KNeighborsClassifier()
clf.fit(X_train, y_train)

y_pred_k= clf.predict(X_test)

print(confusion_matrix(y_test, y_pred_k))

print('accuracy score : ' , accuracy_score(y_test, y_pred_k))

# Support vector classifier

svm_clf= SVC().fit(X_train, y_train)

y_pred_s= svm_clf.predict(X_test)


print(confusion_matrix(y_test, y_pred_s))

print('accuracy score : ' , accuracy_score(y_test, y_pred_s))


# Decision Tree Classifier

tree= DecisionTreeClassifier().fit(X_train, y_train)

y_pred_d= tree.predict(X_test)

print(confusion_matrix(y_test, y_pred_d))

print('accuracy score : ' , accuracy_score(y_test, y_pred_d))

# logistic regression achieved higher accuracy compared to 
# other three algorithms.