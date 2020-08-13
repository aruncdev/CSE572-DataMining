import pandas as pd
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial as poly
import numpy as np

#Data Pre Processing

timeStamp = pd.read_csv("CGMDatenumLunchPat2.csv")
glucoseLevel = pd.read_csv("CGMSeriesLunchPat2.csv")

y = []

for row in range(0,30):
    y.append(row)

deleteRows = []
#Deleting the rows in the csv file if number of data points are less than 26
for row in range(0, 31):
    numericValuesInRow = 0
    for column in range(0, 30):
        if glucoseLevel.iloc[row][column] > 0:
            numericValuesInRow = numericValuesInRow + 1
    if numericValuesInRow < 26:
        print("delete this row because of insufficent data in this period")
        print(row)
        deleteRows.append(row)

glucoseLevel = glucoseLevel.drop(deleteRows)
timeStamp = timeStamp.drop(deleteRows)

#Using a local mean logic to fill the null or NaN values in GlucoseLevel DataFrame
nonNanValues = 0
nonNanSum = 0
for row in range(0, 31):
    nonNanSum = 0
    nonNanValues = 0
    for column in range(0, 30):
        if not glucoseLevel.iloc[row][column] > 0:
            if column == 0 or column == 1:
                glucoseLevel.iloc[row][column] = glucoseLevel.iloc[row][column + 1] if glucoseLevel.iloc[row][column + 1] > 0 else glucoseLevel.iloc[row][column + 2]
            elif column == 29 or column == 30:
                glucoseLevel.iloc[row][column] = glucoseLevel.iloc[row][column - 1] if glucoseLevel.iloc[row][column - 1] > 0 else glucoseLevel.iloc[row][column - 2]
            else:
                nonNanValues = nonNanValues + 1 if glucoseLevel.iloc[row][column + 1] > 0 else nonNanValues
                nonNanSum = nonNanSum + glucoseLevel.iloc[row][column + 1] if glucoseLevel.iloc[row][column + 1] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if glucoseLevel.iloc[row][column + 2] > 0 else nonNanValues
                nonNanSum = nonNanSum + glucoseLevel.iloc[row][column + 2] if glucoseLevel.iloc[row][column + 2] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if glucoseLevel.iloc[row][column - 1] > 0 else nonNanValues
                nonNanSum = nonNanSum + glucoseLevel.iloc[row][column - 1] if glucoseLevel.iloc[row][column - 1] > 0 else nonNanSum
                nonNanValues = nonNanValues + 1 if glucoseLevel.iloc[row][column - 2] > 0 else nonNanValues
                nonNanSum = nonNanSum + glucoseLevel.iloc[row][column - 2] if glucoseLevel.iloc[row][column - 2] > 0 else nonNanSum
                glucoseLevel.iloc[row][column] = round(nonNanSum / nonNanValues,2)
    
#Pre processing of data is completed  


#Extracting features - 1st => Zero Crossing Amplitude difference and Velocity
zeroCrossing = []

for row in range(0, 30):
    temp = []
    for column in range(0, 30):
        temp.append(glucoseLevel.iloc[row][column + 1] - glucoseLevel.iloc[row][column])
    tempMax = max(np.array(temp))
    tempMin = min(np.array(temp))
    zeroCrossing.append(round(tempMax - tempMin, 2))
    
velocity = []

for row in range(0, 30):
    tempVel = []
    for column in range(1, 30):
        displacement = glucoseLevel.iloc[row][column] - glucoseLevel.iloc[row][column - 1]
        #displacement = displacement if displacement > 0 else displacement * (-1)
        tempVel.append(round((displacement / (0.4166)), 2)) # each interval is of 5 min which will be 0.4166 hours
        velocity.append(round(np.mean(tempVel), 2))
# =============================================================================
#     plt.plot(timeStamp.iloc[row][:29], tempVel)
#     plt.title("Velocity Curve")
#     plt.xlabel("Time Stamp")
#     plt.ylabel("Velocity")
# =============================================================================
    

#Extracting features - 2nd => Max - Min - Mean Values
maxValue = []
minValue = []
meanValue = []

for row in range(0, 1):
    maxValue.append(max(glucoseLevel.iloc[row]))
    minValue.append(min(glucoseLevel.iloc[row]))
    meanValue.append(round(np.mean(glucoseLevel.iloc[row]),2))
# =============================================================================
#     plt.title("Min - Mean - Max")
#     plt.xlabel("time")
#     plt.ylabel("glucose level")
#     plt.axhline(maxValue)
#     plt.axhline(minValue)
#     plt.axhline(meanValue)
#     plt.plot(timeStamp.iloc[row], glucoseLevel.iloc[row])
# =============================================================================
    
    
#Extracting features - 3rd => FFT (Fast Fourier Transform)
import numpy as np
import matplotlib.pyplot as pltt
import scipy.fftpack

N = 30
T = 1.0 / N
fftValuesMax = []
fftValuesSecondMax = []

for row in range(0, 30):
    y = glucoseLevel.iloc[row][:30]
    yf = scipy.fftpack.fft(np.array(y))
    xf = np.linspace(timeStamp.iloc[row][0], timeStamp.iloc[row][29], N)
    amplitudes = 2.0/N * np.abs(yf[:N//2])
    amplitudes.sort()
    fftValuesSecondMax.append(round(amplitudes[12],2))
    fftValuesMax.append(round(amplitudes[13],2))
    #plt.plot(xf[15:], amplitudes)

#Extracting features - 4th => Entropy - To measure randomness
import scipy.stats as scipy

entropy = []

for row in range(0, 30):
    entropy_glucoseLevel = glucoseLevel.iloc[row][0:30]
    temp = scipy.entropy(entropy_glucoseLevel)
    entropy.append(temp)

#Extracting features - 5th => RMS - Root Mean Square
import math

rms = []

for row in range(0, 30):
    temp = math.sqrt(sum(value ** 2 for value in glucoseLevel.iloc[row][0:30]) / 30)
    rms.append(round(temp,2))
    
#Creating Feature Matrix

featureMatrixData = pd.DataFrame(columns=[ 'Feat_ZeroCrossing', 'Feat_Velocity', 'Feat_Max', 'Feat_Min', 'Feat_Mean', 'Feat_FFT1',
        'Feat_FFT2', 'Feat_Entropy', 'Feat_RMS' ])
    
for row in range(0, 30):
        featureMatrixData = featureMatrixData.append({
            'Feat_ZeroCrossing': zeroCrossing[row],
            'Feat_Velocity': velocity[row],
            'Feat_Max': maxValue[row],
            'Feat_Min': minValue[row],
            'Feat_Mean': meanValue[row],
            'Feat_FFT1': fftValuesMax[row],
            'Feat_FFT2': fftValuesSecondMax[row],
            'Feat_Entropy': entropy[row],
            'Feat_RMS': rms[row],
        }, ignore_index=True)
    
#PCA => Principal Component Analysis => For reducing dimensions
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # To scale data before performing PCA

X = featureMatrixData[['Feat_ZeroCrossing', 'Feat_Velocity', 'Feat_Max', 'Feat_Min', 'Feat_Mean', 'Feat_FFT1',
        'Feat_FFT2', 'Feat_Entropy', 'Feat_RMS']] # feature vectors

X_std = StandardScaler().fit_transform(X) #Re-scaling

features = X_std.T
covarianceMatrix = np.cov(features) #Co-variance Matrix of features

eigenValues, eigenVectors = np.linalg.eig(covarianceMatrix) #Calculating eigen values and eigen vectors

pcaMatrix = X_std.dot(eigenVectors[:,0:5])

y = []

for val in range(0,30):
    y.append(val)

plt.scatter(y,pcaMatrix[:,0],color = 'red')
plt.scatter(y,pcaMatrix[:,1],color = 'green')
plt.scatter(y,pcaMatrix[:,2],color = 'blue')
plt.scatter(y,pcaMatrix[:,3],color = 'yellow')
plt.scatter(y,pcaMatrix[:,4],color = 'aquamarine')


# =============================================================================
# #PCA Test => Method 2 - Using sklearn package
# from sklearn import decomposition
# 
# pca = decomposition.PCA(n_components = 5)
# sklearn_pca_x = pca.fit_transform(X_std)
# 
# sklearn_result = pd.DataFrame(sklearn_pca_x)
# 
# pcaResult = np.array(sklearn_result)
# 
# plt.scatter(y,pcaResult[:,0],color = 'red')
# plt.scatter(y,pcaResult[:,1],color = 'green')
# plt.scatter(y,pcaResult[:,2],color = 'black')
# plt.scatter(y,pcaResult[:,3],color = 'yellow')
# plt.scatter(y,pcaResult[:,4],color = 'aquamarine')
# =============================================================================

# =============================================================================
# #without rescaling
# 
# pca = decomposition.PCA(n_components = 5)
# matrix = pca.fit_transform(featureMatrixData)
# 
# result = pd.DataFrame(matrix)
# 
# pcaResultWithoutScaling = np.array(result)
# 
# plt.scatter(y,pcaResultWithoutScaling[:,0],color = 'red')
# plt.scatter(y,pcaResultWithoutScaling[:,1],color = 'green')
# plt.scatter(y,pcaResultWithoutScaling[:,2],color = 'black')
# plt.scatter(y,pcaResultWithoutScaling[:,3],color = 'yellow')
# plt.scatter(y,pcaResultWithoutScaling[:,4],color = 'aquamarine')
# =============================================================================
