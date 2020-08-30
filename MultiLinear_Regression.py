import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

#---------------------------Functions------------------

#theetaVector
def theetaMatrix(totalFeatures):
    theetaVectorArray = np.random.uniform(0,2,size = totalFeatures)
    
    return theetaVectorArray

#Normalization
def normalizationFunction(x):
    mean = np.mean(x,axis = 0)
    print("Mean: ",mean)
    standardDeviation = np.std(x) 
    print("StandardDeviation: ",standardDeviation)
    normalizedFeatureArray = x

    normalizedFeatureArray = (normalizedFeatureArray - mean)/standardDeviation
    return normalizedFeatureArray


#Remove Outliers
def removeOutliers(x):
    twoTimesStd = (np.std(x))*2
    mean = np.mean(x,axis = 0)
    
    newArray = np.where(x > (mean + twoTimesStd),(mean + twoTimesStd),x)
    finalArray = np.where(x < (mean - twoTimesStd),(mean - twoTimesStd),x)

    return finalArray

#hypothesis
def hypothesis(theetaArray):
    return np.matmul(featureMatrix,theetaArray)    
    
def partialDerivativeTheetaZero(updatedTheetas):
    newArray = (hypothesis(updatedTheetas) - fareDataY)
    
    return np.sum(newArray)/featureMatrixRows

def partialDerivativesOfTheetas(updatedTheetas,currentFeatureCoefficient):
    newArray = (hypothesis(updatedTheetas) - fareDataY)*(currentFeatureCoefficient)
      
    return np.sum(newArray)/featureMatrixRows
    


def gradientDescent(stepSize):
    a = 0
    theetaArray = theetaMatrix(4)
    tempTheetaArray = theetaArray
    
    while True:
        theetaArray[0] = theetaArray[0] - (stepSize*partialDerivativeTheetaZero(tempTheetaArray))

        for i in range(1,featureMatrixColumns):
            theetaArray[i] = theetaArray[i]- (stepSize*partialDerivativesOfTheetas(tempTheetaArray,featureMatrix[:,i]))
            
        tempTheetaArray = theetaArray
        
        if(a == 1000000):
            return theetaArray
        a+=1
        
            
#--------------------Data Processing--------------------------------------------

dataSet = pd.read_csv("C:/Users/Ali Raza/Google Drive/Logo Generator Project/DataSets/train.csv")
#print(dataSet)

#Age,Sex,Survived features and predict fare
ageDataX = dataSet.iloc[:,5]
ageDataX = np.where(np.isnan(ageDataX), np.ma.array(ageDataX,mask = np.isnan(ageDataX)).mean(axis = 0), ageDataX)
ageDataX = np.reshape(ageDataX,(-1,1))
sexDataX = dataSet.iloc[:,4] #create dummy
survivalDataX = dataSet.iloc[:,1] #dummy
survivalDataX = np.asarray(survivalDataX,dtype = int)
survivalDataX = np.reshape(survivalDataX,(-1,1))
fareDataY = dataSet.iloc[:,9]
#LabelEncoding
lb = LabelEncoder()
sexDataX = lb.fit_transform(sexDataX)
sexDataX = np.reshape(sexDataX,(-1,1))


#Normalization and Outliers Removal
ageDataX = normalizationFunction(ageDataX)
ageDataX = removeOutliers(ageDataX)
fareDataY = normalizationFunction(fareDataY)
fareDataY = removeOutliers(fareDataY)



#featureMatrix
featureShape = ageDataX.shape
featureMatrixRows = featureShape[0]
featureMatrixColumns = featureShape[1]
residueColumn = np.ones((featureMatrixRows,1))
featureMatrix = np.concatenate((residueColumn,ageDataX,sexDataX,survivalDataX),axis = 1)


theetas = gradientDescent(0.0001)
predictedFare = hypothesis(theetas)
print(predictedFare)

