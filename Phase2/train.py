# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:29:36 2020

@author: arunraj
"""

import pandas as pd
import numpy as np 
import math  
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import precision_score,recall_score
import pickle
import scipy.fftpack
from sklearn import svm

meal_data1 = pd.read_csv('mealData1.csv',names=list(range(30)))
meal_data2 = pd.read_csv('mealData2.csv',names=list(range(30)))
meal_data3 = pd.read_csv('mealData3.csv',names=list(range(30)))
meal_data4 = pd.read_csv('mealData4.csv',names=list(range(30)))
meal_data5 = pd.read_csv('mealData5.csv',names=list(range(30)))

meal_data = pd.concat([meal_data1, meal_data2, meal_data3, meal_data4, meal_data5])

meal_data = np.array(meal_data)
meal_data = pd.DataFrame(meal_data)

deleteRows = []
#Deleting the rows in the csv file if number of data points are less than 27
for row in range(0, len(meal_data)):
    numericValuesInRow = 0
    for column in range(0, 30):
        if meal_data.iloc[row][column] > 0:
            numericValuesInRow = numericValuesInRow + 1
    if numericValuesInRow < 27:
        deleteRows.append(row)
        
meal_data = meal_data.drop(deleteRows)

#Using a local mean logic to fill the null or NaN values in meal DataFrame
nonNanValues = 0
nonNanSum = 0
for row in range(0, len(meal_data)):
    nonNanSum = 0
    nonNanValues = 0
    for column in range(0, 30):
        if not meal_data.iloc[row][column] > 0:
            if column == 0 or column == 1:
                meal_data.iloc[row][column] = meal_data.iloc[row][column + 1] if meal_data.iloc[row][column + 1] > 0 else meal_data.iloc[row][column + 2]
            elif column == 29 or column == 30:
                meal_data.iloc[row][column] = meal_data.iloc[row][column - 1] if meal_data.iloc[row][column - 1] > 0 else meal_data.iloc[row][column - 2]
            else:
                nonNanValues = nonNanValues + 1 if meal_data.iloc[row][column + 1] > 0 else nonNanValues
                nonNanSum = nonNanSum + meal_data.iloc[row][column + 1] if meal_data.iloc[row][column + 1] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if meal_data.iloc[row][column + 2] > 0 else nonNanValues
                nonNanSum = nonNanSum + meal_data.iloc[row][column + 2] if meal_data.iloc[row][column + 2] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if meal_data.iloc[row][column - 1] > 0 else nonNanValues
                nonNanSum = nonNanSum + meal_data.iloc[row][column - 1] if meal_data.iloc[row][column - 1] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if meal_data.iloc[row][column - 2] > 0 else nonNanValues
                nonNanSum = nonNanSum + meal_data.iloc[row][column - 2] if meal_data.iloc[row][column - 2] > 0 else nonNanSum
                meal_data.iloc[row][column] = round(nonNanSum / nonNanValues,2)
                
                
#No-meal data pre-processing
                
nomeal_data1 = pd.read_csv('Nomeal1.csv',names=list(range(30)))
nomeal_data2 = pd.read_csv('Nomeal2.csv',names=list(range(30)))
nomeal_data3 = pd.read_csv('Nomeal3.csv',names=list(range(30)))
nomeal_data4 = pd.read_csv('Nomeal4.csv',names=list(range(30)))
nomeal_data5 = pd.read_csv('Nomeal5.csv',names=list(range(30)))

nomeal_data = pd.concat([nomeal_data1, nomeal_data2, nomeal_data3, nomeal_data4, nomeal_data5])

nomeal_data = np.array(nomeal_data)
nomeal_data = pd.DataFrame(nomeal_data)

deleteRows = []
#Deleting the rows in the csv file if number of data points are less than 27
for row in range(0, len(nomeal_data)):
    numericValuesInRow = 0
    for column in range(0, 30):
        if nomeal_data.iloc[row][column] > 0:
            numericValuesInRow = numericValuesInRow + 1
    if numericValuesInRow < 27:
        deleteRows.append(row)
        
nomeal_data = nomeal_data.drop(deleteRows)

#Using a local mean logic to fill the null or NaN values in nomeal DataFrame
nonNanValues = 0
nonNanSum = 0
for row in range(0, len(nomeal_data)):
    nonNanSum = 0
    nonNanValues = 0
    for column in range(0, 30):
        if not nomeal_data.iloc[row][column] > 0:
            if column == 0 or column == 1:
                nomeal_data.iloc[row][column] = nomeal_data.iloc[row][column + 1] if nomeal_data.iloc[row][column + 1] > 0 else nomeal_data.iloc[row][column + 2]
            elif column == 29 or column == 30:
                nomeal_data.iloc[row][column] = nomeal_data.iloc[row][column - 1] if nomeal_data.iloc[row][column - 1] > 0 else nomeal_data.iloc[row][column - 2]
            else:
                nonNanValues = nonNanValues + 1 if nomeal_data.iloc[row][column + 1] > 0 else nonNanValues
                nonNanSum = nonNanSum + nomeal_data.iloc[row][column + 1] if nomeal_data.iloc[row][column + 1] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if nomeal_data.iloc[row][column + 2] > 0 else nonNanValues
                nonNanSum = nonNanSum + nomeal_data.iloc[row][column + 2] if nomeal_data.iloc[row][column + 2] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if nomeal_data.iloc[row][column - 1] > 0 else nonNanValues
                nonNanSum = nonNanSum + nomeal_data.iloc[row][column - 1] if nomeal_data.iloc[row][column - 1] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if nomeal_data.iloc[row][column - 2] > 0 else nonNanValues
                nonNanSum = nonNanSum + nomeal_data.iloc[row][column - 2] if nomeal_data.iloc[row][column - 2] > 0 else nonNanSum
                nomeal_data.iloc[row][column] = round(nonNanSum / nonNanValues,2)


meal_data = meal_data.fillna(meal_data.mean())
nomeal_data = nomeal_data.fillna(nomeal_data.mean())

#Pre- Processing is completed.

#Features

#1 - Zero - Crossing
def ZeroCrossing(data):
    temp = []
    for column in range(0, 26):
        temp.append(data.iloc[column + 1] - data.iloc[column])
        tempMax = max(np.array(temp))
        tempMin = min(np.array(temp))
    return round(tempMax - tempMin, 2)

#2 - Velocity
def Velocity(data):
    temp = []
    for column in range(1, 26):
        displacement = data.iloc[column] - data.iloc[column - 1]
        temp.append(round((displacement / (0.4166)), 2)) # each interval is of 5 min which will be 0.4166 hours
    return round(np.mean(temp), 2)
    
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
    

#Creation feature matrix
def Feature(data):
    feat_matrix = pd.DataFrame(columns = [ 'Velocity', 'Max_Value', 'Min_Value', 'Mean_Value', 'Entropy', 'RMS', 'Min_Max'])
    
    for i in range(0, len(data) - 1):
        feat_matrix = feat_matrix.append({
            'Velocity': Velocity((data.iloc[i][:29])),
            'Max_Value': MaxValue((data.iloc[i][:29])),
            'Min_Value': MinValue((data.iloc[i][:29])),
            'Mean_Value': MeanValue((data.iloc[i][:29])),
            'Entropy': Entropy((data.iloc[i][:29])),
            'RMS': RMS((data.iloc[i][:29])),
            'Min_Max': max(data.iloc[i][:29])-min(data.iloc[i][:29])
            }, ignore_index = True)
    
    return feat_matrix
            
#Feature matrix of Meal and Nomeal Data
 
meal_data = np.array(meal_data)
meal_data = pd.DataFrame(meal_data)
meal_feat_matrix = Feature(meal_data)

nomeal_data = np.array(nomeal_data)
nomeal_data = pd.DataFrame(nomeal_data)
nomeal_feat_matrix = Feature(nomeal_data) 

#Re-scaling 
sc=StandardScaler()
meal_training = sc.fit_transform(meal_feat_matrix)
nomeal_training = sc.fit_transform(nomeal_feat_matrix)

#PCA
final_PCA = PCA(n_components = 5)
final_PCA.fit(meal_training)


pkl_fileName = "PCA.pkl"
with open(pkl_fileName, 'wb') as file:
    pickle.dump(final_PCA, file)
    
transform_PCA = final_PCA.fit_transform(meal_training)
PCA_meal_output = pd.DataFrame(transform_PCA)

transform_PCA = final_PCA.fit_transform(nomeal_training)
PCA_nomeal_output = pd.DataFrame(transform_PCA)

PCA_meal_output['class'] = 1
PCA_nomeal_output['class'] = 0
final_data = PCA_meal_output.append([PCA_nomeal_output])

x_train = final_data.iloc[:,:-1]
y_train = final_data.iloc[:,-1]

k_fold = KFold(5, True, 1)
X = x_train
Y = y_train

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state = 1)
mlp.fit(X, Y)

pkl_fileName = "multiLayerPerceptron_model.pkl"
with open(pkl_fileName, 'wb') as file:
    pickle.dump(mlp, file)

print(mlp.score(X,Y))

for train,test in k_fold.split(X, Y):
    
     x_train, x_test = X.iloc[train], X.iloc[test]
     y_train, y_test = Y.iloc[train], Y.iloc[test]
     
     mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)
     mlp.fit(x_train, y_train)
     print(mlp.score(x_test,y_test))
