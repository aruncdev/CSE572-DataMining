# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:47:12 2020

@author: arunraj
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from train import Feature

file_name = input ("Enter the file name: ")

def Testing(File_name):
    
    with open("multiLayerPerceptron_model.pkl", 'rb') as file:
        multiLayerPerceptron_model = pickle.load(file) 
        Testing_Data = pd.read_csv(File_name, header=None)
    
    with open("PCA.pkl", 'rb') as file:
        PCA = pickle.load(file)
        
    
    Processed_Data = Feature(Testing_Data)
    sc=StandardScaler()
    final_d = sc.fit_transform(Processed_Data)
    final_d=PCA.fit_transform(final_d)
        
    multiLayerPerceptron_model_pred = multiLayerPerceptron_model.predict(final_d)
    print(multiLayerPerceptron_model_pred)
    np.savetxt("classes.csv", multiLayerPerceptron_model_pred, delimiter=",", fmt='%d')
    
Testing(file_name)