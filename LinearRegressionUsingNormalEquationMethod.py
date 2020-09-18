import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder


dataSet = pd.read_csv("C:/Users/Ali Raza/Google Drive/Logo Generator Project/DataSets/train.csv")
#print(dataSet)

#Age,Sex,Survived features and predict fare
ageDataX = dataSet.iloc[:,5]
ageDataX = np.where(np.isnan(ageDataX), np.ma.array(ageDataX,mask = np.isnan(ageDataX)).mean(axis = 0), ageDataX)
ageDataX = np.reshape(ageDataX,(-1,1))
sexDataX = dataSet.iloc[:,4]
survivalDataX = dataSet.iloc[:,1] 
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
featureMatrixColumns = featureShape[1]
residueColumn = np.ones((featureMatrixRows,1))
featureMatrix = np.concatenate((residueColumn,ageDataX,sexDataX,survivalDataX),axis = 1)

def normalEquationMethod(featureMatrix,predictionVector):
    transposeMultiplication = np.matmul(np.transpose(featureMatrix),featureMatrix)
    try:
        inverse = np.linalg.inv(featureMatrix)
    except np.linalg.LinAlgError:
        print("Feature Matrix is not invertible")
        quit()

    theetaArray = np.matmul(inverse,np.transpose(featureMatrix))
    theetaArray = np.matmul(theetaArray,predictionVector)
    return theetaArray

def hypothesis(theetaArray):
    return np.matmul(featureMatrix,theetaArray)

theetas = normalEquationMethod(featureMatrix,fareDataY)
values = hypothesis(theetas)
print(values) 
print(fareDataY)