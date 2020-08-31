#beta version
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataSet = pd.read_csv("C:/Users/Ali Raza/Google Drive/Logo Generator Project/DataSets/train.csv")

ageTrainX = dataSet.iloc[:800,5]
ageTrainX = np.where(np.isnan(ageTrainX), np.ma.array(ageTrainX,mask = np.isnan(ageTrainX)).mean(axis = 0), ageTrainX)
s = ageTrainX.shape
ageTrainXRows = s[0]

#print(np.amin(ageTrainX))
#ageTrainX = normalizationFunction(ageTrainX)

fareTrainY = dataSet.iloc[:800,9]
fareTrainY = np.where(np.isnan(fareTrainY), np.ma.array(fareTrainY,mask = np.isnan(fareTrainY)).mean(axis = 0), fareTrainY)
#fareTrainY = normalizationFunction(fareTrainY)

#ageData = dataSet.iloc[661:,5]
#farePred = dataSet.iloc[661:,9] 

#LINEAR-REGRESSION---------------

#normalizationFunction
def normalizationFunction(x):
    mean = np.mean(x,axis = 0)
    normalizedFeatureArray = np.empty((0,0))
    standardDeviation = np.std(x)
    
    for i in x:
        val = (i - mean)/standardDeviation
        normalizedFeatureArray = np.append(normalizedFeatureArray, val)
    #print(normalizedFeatureArray.shape)
    
    return normalizedFeatureArray

#removeOultiersFunction
def removeOultiers(x):
    std = np.std(x)
    #print("std: ",std)
    mean = np.mean(x,axis = 0)
    #print("mean",mean)
    twoTimeStd = 2*std
    #print("2std", twoTimeStd)
    #print("LA",(twoTimeStd + mean))
    array = np.where(x > (twoTimeStd + mean),(twoTimeStd + mean),x)
    newArray = np.where(array < (mean - twoTimeStd),(mean - twoTimeStd),array)
    
    return newArray

#hypothesis function
def hypothesis(x,alphaZero,alphaOne):
    y = alphaZero + (alphaOne*x)
    
    return y

#costFunction
def costFunction(aZero,aOne):
    hypothesisArray = np.empty((0,0))
    for i in ageTrainX:
        hypothesisArray = np.append(hypothesisArray,hypothesis(i,aZero,aOne))
    
    temp1 = 0
    for m in range(ageTrainXRows):
        temp = (hypothesisArray[m]-fareTrainY[m])**2
        temp1 += temp
        
    return((temp1)/(2*ageTrainXRows))

def partialDerivativeTheetaZero(aZero,aOne):
    hypothesisArray = np.empty((0,0))
    for i in ageTrainX:
        hypothesisArray = np.append(hypothesisArray,hypothesis(i,aZero,aOne))
    
    temp1 = 0
    for m in range(ageTrainXRows):
        temp = hypothesisArray[m] - fareTrainY[m]
        temp1 += temp    
    
    return (temp1/ageTrainXRows)

def partialDerivativeTheetaOne(aZero,aOne):
    hypothesisArray = np.empty((0,0))
    for i in ageTrainX:
        hypothesisArray = np.append(hypothesisArray,hypothesis(i,aZero,aOne))
    
    temp1 = 0
    for m in range(ageTrainXRows):
        temp = (hypothesisArray[m] - fareTrainY[m])*(ageTrainX[m])
        temp1 += temp
        
    return (temp1/ageTrainXRows)

#gradient descent

def gradientDescent(stepSize):
    i = 0
    tempZero = 0.1      #Set values of temps and alphas same
    tempOne = 0.1
    alphaZero = 0.1
    alphaOne = 0.1
    
    while True:   
        alphaZero = alphaZero - (stepSize*partialDerivativeTheetaZero(tempZero, tempOne))
        alphaOne = alphaOne - (stepSize*partialDerivativeTheetaOne(tempZero, tempOne))
        
        #Terminating Condition
        if(partialDerivativeTheetaOne(alphaZero, alphaOne) < 0.1 and partialDerivativeTheetaZero(alphaZero, alphaOne) < 0.1):
            return alphaZero,alphaOne
    
        tempZero = alphaZero 
        tempOne = alphaOne
        
        '''if(i == 100):
            return alphaZero,alphaOne''' #if iterations are needed




#------------Driver Code------------

ageTrainX = normalizationFunction(ageTrainX)
ageTrainX = removeOultiers(ageTrainX)

fareTrainY = normalizationFunction(fareTrainY)
fareTrainY = removeOultiers(fareTrainY)

theetaZero,theetaOne = gradientDescent(0.0002) #StepSize or Learning Rate = 0.0002
print("theetaZero: ",theetaZero)
print("theetaOne", theetaOne)

predictedFare = np.zeros((ageTrainXRows,))

for i in range(ageTrainXRows):
    predictedFare[i] = hypothesis(ageTrainX[i],theetaZero,theetaOne)

plt.scatter(ageTrainX,fareTrainY,color = "red")
plt.plot(ageTrainX,predictedFare,color = "blue")
plt.show()

#print(predictedFare)
#hypothesis(35, theetaZero, theetaOne)



