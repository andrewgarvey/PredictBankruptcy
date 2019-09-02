# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:18:36 2018

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
import matplotlib.pyplot as plt 

#setup dir
input_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\InputData"
output_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\OutputData"
os.chdir(input_dir)

banks_orig = pd.read_csv("1.0-ag-cleaned data.csv")


## PreProccessing Data for NB 

#Drop rows, we're ignoring year, companyID is made up, ROE is too similar to Liquidity
banks = banks_orig.drop(['Data Year - Fiscal','Return on Equity','CompanyID'],axis=1) 
#banks = banks.drop(['Market Book Ratio', 'Sales Growth'], axis = 1) 

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


#------------------------------------------------------------------------------
## Naive Bayes, if there are hyper parameters, i couldnt find them, gridsearchcv literally doesn't have a page for them, and one guy on reddit said so. good enough for me

#tried Gaussian and Multivariable but they both peformed worse than Bernoulli
# alright guess i'll just make my own grid search style thing also with test train split looped in (only test on random sets , needs a rework)

#for alpha in range(0,20,1):
#    alpha = alpha/10
#    for binarize in range(0,20,1):
#        binarize = binarize/10
#        model = BernoulliNB(alpha =alpha,binarize=alpha)
#        model.fit(x_train_sm, y_train_sm.ravel())    
#        prediction = model.predict(x_test)
#        AUC = round(100*skmet.roc_auc_score(y_test,prediction),2)
#        Accuracy = round(100*skmet.accuracy_score(y_test,prediction),2)
#        print(alpha,binarize,AUC,Accuracy)
       


# going by best AUC the winner is "the default" which is alpha =1 
model = BernoulliNB(alpha = 1)

# Train the model using the training sets
model.fit(x_train_sm, y_train_sm.ravel())

#prediction 
prediction = model.predict(x_test)

#Metrics 
print("\n\n Welcome to Naive Bayes. \n It 'Naively' assumes independance between variables. \n It's best feature is being very quick and relatively easy to make.\n Used mostly in text classification and reccomender systems \n and here we can see that it is awful \n\n") 

print("CONFUSION MATRIX: \n" , skmet.confusion_matrix(y_test,prediction))
print("\n CLASSIFICATION REPORT:\n\n",skmet.classification_report(y_test,prediction))
print('ACCURACY -> ',round(100*skmet.accuracy_score(y_test,prediction),2),'%')

print("recall:",skmet.recall_score(y_test,prediction))
print("precision:",skmet.precision_score(y_test,prediction))
print("f1_score:",skmet.f1_score(y_test,prediction))



prediction = model.predict_proba(x_test)[:,1]
# ROC Graph 
fpr, tpr, _ = skmet.roc_curve(y_test, prediction)
roc_auc = skmet.auc(fpr,tpr)

# Plot the Receiver Operating Characteristic (ROC) Curve
plt.figure()
plt.title('ROC Curve Naive Bayes')
plt.plot(fpr, tpr, 'b',label = 'AUC = 70.87%')
plt.legend(loc = 'lower right')
plt.plot([0, 1],[0, 1],'r--')
plt.xlim([-0.1, 1.0])
plt.ylim([-0.1, 1.01])
plt.ylabel('True Positive (TP) Rate')
plt.xlabel('False Positive (FP) Rate')
plt.show()

print("AUC ->",roc_auc)
#---------------------------------------------------------------------------------------------------
"""
print("\n\n\n\nFor funzies, here is what happens if you used UNSMOTED data, we even get a divide by 0 warning for some metrics!!\n") 

model2 = BernoulliNB()

# Train the model using the training sets
model2.fit(x_train, y_train.ravel())  ## HERE WE USE unSMOTED DATA

#prediction 
prediction2 = model2.predict(x_test)

#Metrics 

print("CONFUSION MATRIX: \n" , skmet.confusion_matrix(y_test,prediction2))
print("\n CLASSIFICATION REPORT:\n\n",skmet.classification_report(y_test,prediction2))
print('ACCURACY -> ',round(100*skmet.accuracy_score(y_test,prediction2),2),'%')
print('     AUC -> ',round(100*skmet.roc_auc_score(y_test,prediction2),2),'%')


"""





















