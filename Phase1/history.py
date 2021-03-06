plt.scatter(y,pcaMatrix[:,2],color = 'blue')
plt.scatter(y,pcaMatrix[:,3],color = 'yellow')
plt.scatter(y,pcaMatrix[:,4],color = 'aquamarine')
clear
plt.scatter(y,pcaResult[:,0],color = 'red')
plt.scatter(y,pcaResult[:,1],color = 'green')
plt.scatter(y,pcaResult[:,2],color = 'blue')
plt.scatter(y,pcaResult[:,3],color = 'yellow')
plt.scatter(y,pcaResult[:,4],color = 'aquamarine')
clear
plt.scatter(y,pcaMatrix[:,0],color = 'red')
plt.scatter(y,pcaMatrix[:,1],color = 'green')
plt.scatter(y,pcaMatrix[:,2],color = 'blue')
plt.scatter(y,pcaMatrix[:,3],color = 'yellow')
plt.scatter(y,pcaMatrix[:,4],color = 'aquamarine')


#PCA Test => Method 2 - Using sklearn package
from sklearn import decomposition

pca = decomposition.PCA(n_components = 5)
sklearn_pca_x = pca.fit_transform(X_std)

sklearn_result = pd.DataFrame(sklearn_pca_x)

pcaResult = np.array(sklearn_result)

plt.scatter(y,pcaResult[:,0],color = 'red')
plt.scatter(y,pcaResult[:,1],color = 'green')
plt.scatter(y,pcaResult[:,2],color = 'blue')
plt.scatter(y,pcaResult[:,3],color = 'yellow')
plt.scatter(y,pcaResult[:,4],color = 'aquamarine')
plt.scatter(y,pcaResult[:,0],color = 'red')
plt.scatter(y,pcaResult[:,1],color = 'green')
plt.scatter(y,pcaResult[:,2],color = 'blue')
plt.scatter(y,pcaResult[:,3],color = 'yellow')
plt.scatter(y,pcaResult[:,4],color = 'aquamarine')
plt.scatter(y,pcaMatrix[:,0],color = 'red')
plt.scatter(y,pcaMatrix[:,1],color = 'green')
plt.scatter(y,pcaMatrix[:,2],color = 'blue')
plt.scatter(y,pcaMatrix[:,3],color = 'yellow')
plt.scatter(y,pcaMatrix[:,4],color = 'aquamarine')


#PCA Test => Method 2 - Using sklearn package
from sklearn import decomposition

pca = decomposition.PCA(n_components = 5)
sklearn_pca_x = pca.fit_transform(X_std)

sklearn_result = pd.DataFrame(sklearn_pca_x)

pcaResult = np.array(sklearn_result)

plt.scatter(y,pcaResult[:,0],color = 'red')
plt.scatter(y,pcaResult[:,1],color = 'green')
plt.scatter(y,pcaResult[:,2],color = 'black')
plt.scatter(y,pcaResult[:,3],color = 'yellow')
plt.scatter(y,pcaResult[:,4],color = 'aquamarine')
clear
pca = decomposition.PCA(n_components = 5)
matrix = pca.fit_transform(featureMatrixData)

result = pd.DataFrame(matrix)

pcaResultWithoutScaling = np.array(result)

plt.scatter(y,pcaResultWithoutScaling[:,0],color = 'red')
plt.scatter(y,pcaResultWithoutScaling[:,1],color = 'green')
plt.scatter(y,pcaResultWithoutScaling[:,2],color = 'black')
plt.scatter(y,pcaResultWithoutScaling[:,3],color = 'yellow')
plt.scatter(y,pcaResultWithoutScaling[:,4],color = 'aquamarine')

## ---(Tue Feb 11 06:23:55 2020)---
runfile('C:/Users/arunraj/.spyder-py3/Project1.py', wdir='C:/Users/arunraj/.spyder-py3')
time = pd.read_csv("CGMDatenumLunchPat2.csv")
glucose = pd.read_csv("CGMSeriesLunchPat2.csv")
plt.title("time vs glucose")
plt.xlabel("TIme")
plt.ylabel("Glucose")
plt.plot(timeStamp.iloc[0], glucoseLevel.iloc[0])
plt.plot(time.iloc[0], glucose.iloc[0])
plt.show()
plt.plot(timeStamp.iloc[0], glucoseLevel.iloc[0])
plt.plot(timeStamp.iloc[0], glucoseLevel.iloc[0])
plt.show()
plt.title("time vs glucose")
plt.xlabel("Time")
plt.ylabel("Glucose")
plt.plot(timeStamp.iloc[0], glucoseLevel.iloc[0])
plt.show()
plt.title("time vs glucose")
plt.xlabel("Time")
plt.ylabel("Glucose")
plt.plot(timeStamp[0], glucoseLevel[0])
plt.show()
plt.title("time vs glucose")
plt.xlabel("Time")
plt.ylabel("Glucose")
plt.plot(time.iloc[0], glucose.iloc[0])
plt.show()

## ---(Tue Feb 11 08:14:22 2020)---
y = []
for row in range(0, 30):
    y.appened(row)
y = []
for row in range(0, 30):
    y.append(row)
plt.plot(y, zeroCrossing)
from matplotlib import pyplot as plt
plt.plot(y, zeroCrossing)
zeroCrossing = []

for row in range(0, 30):
    temp = []
    for column in range(0, 30):
        temp.append(glucoseLevel.iloc[row][column + 1] - glucoseLevel.iloc[row][column])
    tempMax = max(np.array(temp))
    tempMin = min(np.array(temp))
    zeroCrossing.append(round(tempMax - tempMin, 2))


y = []
for row in range(0, 30):
    y.append(row)


plt.plot(y, zeroCrossing)
import pandas as pd
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial as poly
import numpy as np

#Data Pre Processing

timeStamp = pd.read_csv("CGMDatenumLunchPat2.csv")
glucoseLevel = pd.read_csv("CGMSeriesLunchPat2.csv")

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


y = []
for row in range(0, 30):
    y.append(row)


plt.plot(y, zeroCrossing)
timeStamp = pd.read_csv("CGMDatenumLunchPat2.csv")
glucoseLevel = pd.read_csv("CGMSeriesLunchPat2.csv")
runfile('C:/Users/arunraj/.spyder-py3/Project1.py', wdir='C:/Users/arunraj/.spyder-py3')
plt.plot(y, zeroCrossing)                                       