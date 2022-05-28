#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:05:54 2022

Script takes the csv files grounded_lineage_2010 and 2012 and reconfigures that same info into a 
different format that is required by other scripts. 


@author: dmueller
"""

import os
import pandas as pd
import numpy as np

# change dir and read in files to numpy arrays
os.chdir('.') #assuming you are in the directory you need to be already... 

ii2010 = np.loadtxt('grounded_lineage_2010.csv', delimiter=',', dtype=str)
ii2012 = np.loadtxt('grounded_lineage_2012.csv', delimiter=',', dtype=str)

# each lineage is in its own column, transpose row wise
ii2010 = ii2010.transpose()
ii2012 = ii2012.transpose()
ncols = max(ii2010.shape[1], ii2012.shape[1])

# pad out columns and concatenate to one array
ii2012 = np.pad(ii2012, [(0,0),(0,ncols-ii2012.shape[1])], 'constant', constant_values='NA')     
ii2010 = np.pad(ii2010, [(0,0),(0,ncols-ii2010.shape[1])], 'constant', constant_values='NA')     
ii = np.concatenate([ii2010,ii2012])

# all NA will be nan
ii[ii=="NA"] = np.nan

# define 2 empty lists
nickname = []
lineage = []

# go to each row... 
for i in range(len(ii)):
    nickname.append(ii[i][0].split('_')[-1])
    lineage.append(list(ii[i][ii[i] != 'nan']))

ii = pd.DataFrame(zip(nickname,lineage), columns=["nickname","lineage"])
ii.to_csv('grounded_lineage.csv',index=False)



