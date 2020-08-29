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
sexDataX = dataSet.iloc[:,4] #create dummy
survivalDataX = dataSet.iloc[:,1] #dummy
fareDataY = dataSet.iloc[:,9]
#LabelEncoding
lb = LabelEncoder()
sexDataX = lb.fit_transform(sexDataX)

#featureMatrix
featureMatrix = np.concatenate()


#---------------------------Functions------------------

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
    return featureArray*theetaArray    

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

