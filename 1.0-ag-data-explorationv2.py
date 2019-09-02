# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:08:54 2018

@author: 9atg
"""

#setup
import os
import pandas as pd 
import numpy as np 


input_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\InputData"
output_dir = "D:\QUEENS MMAI\823 Finance\Assign\Assign1\OutputData"
os.chdir(input_dir)

banks=pd.read_csv('Bankruptcy_data_Final.csv')

#Making the calculation efficient time-wise
CompanyID=np.zeros(len(banks))
year=banks.loc[:,'Data Year - Fiscal']

# Making Company ID
id=1
for i in range(0,len(banks)):
    if (i < len(banks)-1):
        firstvalue = year[i]
        secondvalue= year[i+1]
        CompanyID[i] = id 
        if (secondvalue!=firstvalue+1): #this still won't get companies that fluke out and have A -> 2000-2005 and then B -> 2006-2010, as far as i can tell, nothing reasonably could 
            id = id + 1
    else:
        CompanyID[i]= id #catches the last one that would be out of index


banks['CompanyID']=CompanyID # add it to main df

# Filling out the missings, i choose to forwardfill or backfill,far more accurate than average when dealing with many many years (would still return nan, if it only has nan)

final=pd.DataFrame()
for i in range(0,len(np.unique(CompanyID))): # takes a few minutes
    newdf = banks.loc[CompanyID==i+1,:]  # just grab company i
    newdf = newdf.ffill().bfill() #ffill then bfill 
    final = final.append(newdf)    #append it all into 1 thing


# Check up on nan
final.isnull().sum() 

# Dropping still remaining rows with nan,  (this means a company NEVER reported any stats for an entire column), may be a better way but for this i'm ok
final2 = final.dropna()

# Check up on nan
final2.isnull().sum() 

#write csv
os.chdir(output_dir)
final2.to_csv("1.0-ag-cleaned data.csv", index= False ) # I kept everything, model to your hearts content!  
