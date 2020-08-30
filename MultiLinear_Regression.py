import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder

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

#featureMatrix
featureShape = ageDataX.shape
featureMatrixRows = featureShape[0]
residueColumn = np.ones((featureMatrixRows,1))
featureMatrix = np.concatenate((residueColumn,ageDataX,sexDataX,survivalDataX),axis = 1)

#---------------------------Functions------------------

#theetaVector
def theetaMatrix(totalFeatures):
    theetaVectorArray = np.random.uniform(0,2,size = totalFeatures)
    
    return theetaVectorArray

theetaVector = theetaMatrix(4)

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
def hypothesis(featureArray, theetaArray):
    newArray = featureArray*theetaArray    
    return np.sum(newArray)
    
def partialDerivativeTheetaZero(featureArray,updatedTheetas):
    hypothesisArray = featureArray*updatedTheetas
    return  (hypothesisArray - fareDataY)/featureArrayRows  #Define ROws

def partialDerivativesOfTheetas(updatedTheetas,currentFeatureCoefficient):
    hypothesisArray = featureArray*updatedTheetas
    return  ((hypothesisArray - fareDataY)*currentFeatureCoefficient)/featureArrayRows

    


def gradientDescent (stepSize):

    while True:
        theetaZero = theetaZero - (stepSize*partialDerivativeTheetaZero(arguements))

        for i in theetaArray:
            updatedTheetas = np.append(updatedTheetas,partialDerivativesOfTheetas(updatedTheetas,i))

