# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:32:58 2018
For Learning Mostly
@author: 9atg
"""

#setup basic
import os
import pandas as pd 
import numpy as np 


#setup NB required things 
import random
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmet

#Setup, extra RF required things 
from sklearn.ensemble import RandomForestClassifier 


#setup dir
input_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\InputData"
output_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\OutputData"
os.chdir(input_dir)

banks_orig = pd.read_csv("1.0-ag-cleaned data.csv")


## PreProccessing Data for RF

#Drop rows, we're ignoring year, companyID is made up, ROE is too similar to Liquidity
banks = banks_orig.drop(['Data Year - Fiscal','Return on Equity','CompanyID'],axis=1) 

#split x and Y 
x_full = banks.drop('BK',axis =1)
y_full = banks.loc[:,['BK']]

#Normalize Data (NOT BK)
x_normalized =pd.DataFrame(preprocessing.scale(x_full))
full_normalized = pd.concat([x_normalized,y_full],axis=1)

##test train sets
train,test = train_test_split(full_normalized,test_size=0.2,random_state=123)

x_train = np.array(train.drop('BK',axis =1))  #needed as arrays so that we can "ravel"
y_train = np.array(train.loc[:,['BK']]) 

x_test = np.array(test.drop('BK',axis =1))
y_test = np.array(test.loc[:,['BK']])

#Smote Data 
sm = SMOTE(random_state=123) 
x_train_sm, y_train_sm = sm.fit_sample(x_train,y_train.ravel())

##--------------------------------------------------------------------------
#RandomForest 
rfc = RandomForestClassifier(random_state=123)

#parameters
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

#scoring
scoring = {'AUC': 'roc_auc', 'Accuracy': skmet.make_scorer(skmet.accuracy_score)}

#gridsearch
CV_rfc = GridSearchCV(estimator=rfc,param_grid=param_grid, cv= 5, scoring = scoring, refit='AUC', return_train_score=True)

#train 
CV_rfc.fit(x_train_sm, y_train_sm.ravel())

#check best
CV_rfc.best_params_ #used in next line

#use best params 
rfc1=RandomForestClassifier(CV_rfc.best_params_)

#re-train using best params
rfc1.fit(x_train_sm, y_train_sm.ravel())


#predict it 
prediction=rfc1.predict(x_test)

#Metrics 

print("CONFUSION MATRIX: \n" , skmet.confusion_matrix(y_test,prediction))
print("\n CLASSIFICATION REPORT:\n\n",skmet.classification_report(y_test,prediction))
print('ACCURACY -> ',round(100*skmet.accuracy_score(y_test,prediction),2),'%')
print('     AUC -> ',round(100*skmet.roc_auc_score(y_test,prediction),2),'%')



