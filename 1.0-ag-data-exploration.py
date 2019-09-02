# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:25:52 2018

@author: Andrew Garvey
"""
   
#setup 
import os
import pandas as pd 
import numpy as np 

import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#shamlessly stolen  
def plot_corr(df,size=15):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


# some dir stuff
input_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\InputData"
ouput_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\OutputData"
os.chdir(input_dir)

banks=pd.read_csv('Bankruptcy_data_Final.csv')



#some basic exploring of exploratory stuff 
#plot_corr(banks)

banks.head()
banks.describe()#this doesn't show up in its entirety? normal?
banks.index
banks.columns
banks.values


# CLEANING PLAN 
# 1. Drop any rows that have NaN (ideally we'd impute them)
# 2. Remove the Year (ideally i'd like to shift it by 1 as i think it'd more predictive, or atleast more interesting)
# 3. Oversample (ideally we'd SMOTE)


# 1.Drop entire rows with any NaN
banks1 = banks.dropna()


# 2.remove the year 
banks2 = banks1.drop(['Data Year - Fiscal'],axis =1)

# 3.Oversampling process 

# figure out how many 1s and 0s we have right now
number1= len(banks2.loc[banks2['BK']==1])
number0= len(banks2) - number1



# there appear to be 473 1s and  80731 0s, lets add a bunch more 1s VIA duplication, so they have basically the same amount
BK1 = banks2.loc[banks2['BK']==1]
BK0 = banks2.loc[banks2['BK']==0]

#pick a number to make them basically the same
ratio = round(number0/number1) 
DupBK1 = pd.concat([BK1]*ratio)

#Now Union them together
Final = BK0.append(DupBK1)

# double check numbers
FinalNumber1= len(Final.loc[Final['BK']==1])
FinalNumber0= len(Final) - FinalNumber1


#randomly split 
random.seed(123)
train, test = train_test_split(Final, test_size=0.2)

#assigning predictor and target variables
x_train = np.array(train.drop('BK',axis=1))
x_train = preprocessing.scale(x_train)
y_train = train.loc[:,'BK']


x_test = np.array(test.drop('BK',axis=1))
x_test = preprocessing.scale(x_test)
y_test = test.loc[:,'BK']

## Naive Bayes 

model = BernoulliNB()

# Train the model using the training sets 
model.fit(x_train, y_train)

#prediction 
prediction = model.predict(x_test)

#confusion matrix
conf_mat = confusion_matrix(y_test,prediction)
print(conf_mat)


sum(prediction==1)
len(prediction)
sum(y_test)
len(y_test)





























