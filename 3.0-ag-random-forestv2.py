# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 00:20:09 2018

@author: 9atg
"""

#IMPORT 
import os
import pandas as pd 
import numpy as np 
import random
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmet
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skmet
from imblearn.over_sampling import SMOTE
import random
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as skmet
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

#-----------------------------------------------------------------------------------------
#setup dir
input_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\InputData"
output_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\OutputData"
os.chdir(input_dir)

banks_orig = pd.read_csv("1.0-ag-cleaned data.csv")


## PreProccessing Data for NB 

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


#----------------------------------------------------------------------------------------------------
# Making RF this time
model = RandomForestClassifier(random_state=123)

param_grid_val = [{
    'n_estimators': [200],
    'bootstrap': [True],
    'max_depth': [150,100],
    'max_features': ['auto','sqrt'],
    'min_samples_leaf': [150]
}]


clf = GridSearchCV(model, param_grid_val, cv=3, n_jobs=-1, verbose=2)
clf.fit(x_train_sm, y_train_sm)
prediction=clf.predict(x_test)

prediction = clf.predict_proba(x_test)[:,1]



# Confusion matrix 
print("Confusion matrix: \n" ,skmet.confusion_matrix(y_test,prediction))
# Metrics
print("Accuracy:",skmet.accuracy_score(y_test, prediction))
print("recall:",skmet.recall_score(y_test, prediction))
print("precision:",skmet.precision_score(y_test,prediction))
print("f1_score:",skmet.f1_score(y_test, prediction))
print("roc_auc:",skmet.roc_auc_score(y_test,prediction))


# Plot ROC, 

# shamelessly stolen from neal
# ROC Graph 
fpr, tpr, _ = skmet.roc_curve(y_test, prediction)
roc_auc = skmet.auc(fpr,tpr)

# Plot the Receiver Operating Characteristic (ROC) Curve
plt.figure()
plt.title('ROC Curve Random Forest')
plt.plot(fpr, tpr, 'b',label = 'AUC = 92.82%')
plt.legend(loc = 'lower right')
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive (TP) Rate')
plt.xlabel('False Positive (FP) Rate')
plt.show()


print(roc_auc)




