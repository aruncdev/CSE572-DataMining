# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:59:36 2020

@author: arunraj
"""


import pandas as pd
import numpy as np 
import math  
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.fftpack
import pickle


meal_data1 = pd.read_csv('mealData1.csv',names=list(range(30)))
meal_data2 = pd.read_csv('mealData2.csv',names=list(range(30)))
meal_data3 = pd.read_csv('mealData3.csv',names=list(range(30)))
meal_data4 = pd.read_csv('mealData4.csv',names=list(range(30)))
meal_data5 = pd.read_csv('mealData5.csv',names=list(range(30)))

meal_amount_data1 = pd.read_csv('mealAmountData1.csv', names=[30])
meal_amount_data3 = pd.read_csv('mealAmountData3.csv', names=[30])
meal_amount_data4 = pd.read_csv('mealAmountData4.csv', names=[30])
meal_amount_data2 = pd.read_csv('mealAmountData2.csv', names=[30])
meal_amount_data5 = pd.read_csv('mealAmountData5.csv', names=[30])

meal_data = pd.concat([meal_data1.iloc[:50,:], meal_data2.iloc[:50,:], meal_data3.iloc[:50,:], meal_data4.iloc[:50,:], meal_data5.iloc[:50,:]])
meal_amount_data = pd.concat([meal_amount_data1.iloc[:50,:], meal_amount_data2.iloc[:50,:], meal_amount_data3.iloc[:50,:], meal_amount_data4.iloc[:50,:], meal_amount_data5.iloc[:50,:]])

final_df = pd.concat([meal_data, meal_amount_data], axis=1)

final_array = np.array(final_df)
final_df = pd.DataFrame(final_array)

#Deleting the rows in the data frame if number of data points are less than 27

deleteRows = []

for row in range(0, len(final_df)):
    numericValuesInRow = 0
    for column in range(0, 30):
        if final_df.iloc[row][column] > 0:
            numericValuesInRow = numericValuesInRow + 1
    if numericValuesInRow < 27:
        deleteRows.append(row)
        
final_df = final_df.drop(deleteRows)

#Filling NaN values in the data frame
final_df = final_df.interpolate(axis=1,method='quadratic',limit=10, limit_direction='both')
final_df = final_df.dropna(axis=0,how='any')
final_array = np.array(final_df)
final_df = pd.DataFrame(final_array)

#Pre- processing is completed

#Feature Extraction

def ZeroCrossing(data):
    temp = []
    for column in range(0, 26):
        temp.append(data.iloc[column + 1] - data.iloc[column])
        tempMax = max(np.array(temp))
        tempMin = min(np.array(temp))
    return round(tempMax - tempMin, 2)

#2 - Velocity
def Velocity(data):
    initial = data.iloc[0]
    final = data.iloc[-1]
    displacement = final - initial 
    return round(displacement/7)
    
#3 - Max Value - Min Value - Mean Value
def MaxValue(data):
    return max(data)
def MinValue(data):
    return min(data)
def MeanValue(data):
    return round(np.mean(data),2)

#4 - FFT1 - FFT2
def FFT1(data):
    N = 29
    y = data[:26]
    yf = scipy.fftpack.fft(np.array(y))
    amplitudes = 2.0/N * np.abs(yf[:N//2])
    amplitudes.sort()
    return round(amplitudes[12],2)

def FFT2(data):
    N =29
    y = data[:26]
    yf = scipy.fftpack.fft(np.array(y))
    amplitudes = 2.0/N * np.abs(yf[:N//2])
    amplitudes.sort()
    return round(amplitudes[11],2)
    
#5 - Entropy

def Entropy(data):
    entropy = scipy.stats.entropy(data)
    return entropy

#6 - RMS
def RMS(data):
    rms = math.sqrt(sum(value ** 2 for value in data.iloc[0:26]) / 26)
    return round(rms,2)

# Feature Matrix
from scipy.stats import skew
def Feature(data):
    feat_matrix = pd.DataFrame(columns = [ 'Rise_In_Insulin', 'CGM_Difference', 
                                          'Skewness', 'Variance', 'Max_Value', 'Min_Value', 
                                          'Mean_Value', 'Entropy', 'RMS'])
    
    for i in range(0, len(data)):
        feat_matrix = feat_matrix.append({
            'Rise_In_Insulin':data.iloc[i][9]-data.iloc[i][6]/(45-30),
            'CGM_Difference':(max(data.iloc[i])-data.iloc[i][6])/data.iloc[i][6] if data.iloc[i][6]!=0 else 0,
            'Skewness' : skew(data.iloc[i][:29]), 
            'Variance' : np.var(data.iloc[i][:29]),
            'Max_Value': MaxValue(data.iloc[i][:29]),
            'Min_Value': MinValue((data.iloc[i][:29])),
            'Mean_Value': MeanValue((data.iloc[i][:29])),
            'Entropy': Entropy((data.iloc[i][:29])),
            'RMS': RMS((data.iloc[i][:29])),
            }, ignore_index = True)
    
    return feat_matrix

feature_matrix = Feature(final_df)

#Assigning bins
bins = []

for i in final_df.iloc[:,30]:
    if i == 0:
        bins.append(1)
    elif i > 0 and i < 21:
        bins.append(2)
    elif i > 20 and i < 41:
        bins.append(3)
    elif i > 40 and i < 61:
        bins.append(4)
    elif i > 60 and i < 81:
        bins.append(5)
    elif i > 80 and i < 101:
        bins.append(6)
    else:
        bins.append(7)
    

final_df['assigned_bins'] = bins

#Re-scaling 
sc = StandardScaler()
feature_Scaled = sc.fit_transform(feature_matrix)

#PCA
final_PCA = PCA(n_components = 2)
final_PCA.fit(feature_Scaled)

pkl_fileName = "PCA.pkl"
with open(pkl_fileName, 'wb') as file:
    pickle.dump(final_PCA, file)
    
pca_1 = pd.DataFrame(final_PCA.fit_transform(feature_Scaled))
pca_1.to_csv('temp_pca.csv')

#K MEANS 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

kmeans_cluster = KMeans(n_clusters=6)
cluster_labels = kmeans_cluster.fit_predict(pca_1)
predicted_Labels = list(kmeans_cluster.labels_)

final_df['predictedLabels'] = predicted_Labels 

#Assigning bin value to the cluster
from collections import Counter

binCount={}
for i in range(0, 6):
    temp = final_df[final_df['predictedLabels'] == i]['assigned_bins']
    count = Counter(list(temp))
    binCount[i] = count.most_common(1)[0][0] 


#Assigning cluster bins
final_df['kmeans_clusters']=predicted_Labels
temp = {}
for i in range(0,6,1):
    tempList = final_df[final_df['kmeans_clusters']==i]['assigned_bins']
    counter_list = Counter(list(tempList))
    
    
    common = counter_list.most_common(1)[0][0]
    clusCount=1
    while clusCount < 4 and clusCount < len(counter_list):
            
            if common in temp.values():
               common = (counter_list.most_common(clusCount + 1))[clusCount][0]
            else:
               break
            clusCount = clusCount + 1
    
    temp[i] = common
    
    
final_df['kmeans_bins']=final_df['kmeans_clusters']
final_df['kmeans_bins']=final_df['kmeans_bins'].map(
        {0:temp[0],1:temp[1],2:temp[2],3:temp[3],4:temp[4],5:temp[5]})

#DBSCAN
epsilon= 0.35
min_sample=4
tempResult = DBSCAN(eps = epsilon, min_samples = min_sample)
clusters = tempResult.fit_predict(pca_1)
dbscan_tempResult = pd.DataFrame({'pc1':list(pca_1.iloc[:,0]),'pc2':list(pca_1.iloc[:,1]),'cluster':list(clusters)})
outliers_df = dbscan_tempResult[dbscan_tempResult['cluster'] == -1].iloc[:,0:2]

#Using KMeans to assign bins for outliers
from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=4,p=2)
knn.fit(dbscan_tempResult[dbscan_tempResult['cluster'] != -1].iloc[:,0:2], dbscan_tempResult[dbscan_tempResult['cluster'] != -1].iloc[:,2])
for x,y in zip(outliers_df.iloc[:,0], outliers_df.iloc[:,1]):
    dbscan_tempResult.loc[(dbscan_tempResult['pc1'] == x) & (dbscan_tempResult['pc2'] == y), 'cluster'] = knn.predict([[x,y]])[0]


final_df['dbScan_clusters'] = dbscan_tempResult['cluster']
temp = {}
for i in range(0,6,1):
    tempList = final_df[final_df['dbScan_clusters'] == i]['assigned_bins']
    counter_list=Counter(list(tempList))
    
    
    common = counter_list.most_common(1)[0][0]
    clusCount = 1
    while clusCount < 4 and clusCount < len(counter_list):
            if common in temp.values():
               common = (counter_list.most_common(clusCount + 1))[clusCount][0]
            else:
               break
            clusCount = clusCount + 1
    
    temp[i] = common
    
   
     
final_df['dbscan_bins']=final_df['dbScan_clusters']
final_df['dbscan_bins']=final_df['dbscan_bins'].map(
        {0:temp[0], 1:temp[1], 2:temp[2], 3:temp[3], 4:temp[4], 5:temp[5]})

 
#Resultant Files
train_file = feature_matrix
train_file['bins'] = bins
train_file['kmeans_bins'] = final_df['kmeans_bins']
train_file['dbscan_bins'] = final_df['dbscan_bins']
feature_matrix.to_csv('train_file.csv',index=False)