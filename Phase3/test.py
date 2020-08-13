# -*- coding: utf-8 -*-
"""
Created on Thu May  4 06:06:57 2020

@author: arunraj
"""

import pandas as pd
from train import Feature
from sklearn.neighbors import KNeighborsClassifier

test = pd.read_csv('proj3_test.csv',names=[i for i in range(30)],index_col=False)  
test.fillna(0)   
features = Feature(test)
features = features.fillna(0)  
          

train_features = pd.read_csv('train_file.csv')
train_features = train_features.fillna(4) 
knn = KNeighborsClassifier(n_neighbors=4, p=2)
knn.fit(train_features.iloc[:,:-3], train_features.iloc[:,-2])
knn1 = KNeighborsClassifier(n_neighbors=4, p=2)
knn1.fit(train_features.iloc[:,:-3], train_features.iloc[:,-1])
    
kmeans_Results = []
dbscan_Results = []        

for i in range(0, features.shape[0]):
    kmeans_Results.append(int(knn.predict([list(features.iloc[i,:])])[0]))
    dbscan_Results.append(int(knn1.predict([list(features.iloc[i,:])])[0]))
        

results_df = pd.DataFrame({'db': dbscan_Results, 'k': kmeans_Results})
print(results_df)

results_df.to_csv('result_file.csv',index=False)