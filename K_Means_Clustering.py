import numpy as np
import cv2
import glob
import random
import math
import matplotlib.pyplot as plt


#-------functions-------------
def centroidsDistance(dataMatrixRow,updatedCentroidsArray):
    distances = np.empty((0,))
    dataMatrixRowShape = dataMatrixRow.shape
    updatedCentroidsArrayShape = updatedCentroidsArray.shape
    updatedCentroidsArrayRows = updatedCentroidsArrayShape[0]

    for i in range(updatedCentroidsArrayRows) :
        distances = np.append(distances,(dataMatrixRow[0]-updatedCentroidsArray[i,0])**2 + (dataMatrixRow[1]-updatedCentroidsArray[i,1])**2 + (dataMatrixRow[2]-updatedCentroidsArray[i,2])**2)

    return distances


def updateCentroids(dataMatrixRow,distanceArray,updatedCentroidsArray):
    minimumDistanceCentroid = np.where(distanceArray == (np.amin(distanceArray)))
    updatedCentroidsArray[minimumDistanceCentroid] = (dataMatrixRow + updatedCentroidsArray[minimumDistanceCentroid])/2
    
    return updatedCentroidsArray

def kMeansClustering(dataMatrix, totalCentroids):
    centroidsArray = np.empty((totalCentroids,3))

    for i in range(totalCentroids):
        centroidsArray[i] = dataMatrix[random.randint(0,dataMatrixRows)]
    #print(centroidsArray)
    
    for i in range(dataMatrixRows):
        distanceArray = centroidsDistance(dataMatrix[i,:],centroidsArray)
        centroidsArray = updateCentroids(dataMatrix[i,:],distanceArray,centroidsArray)
    
    return centroidsArray


#-------------Data Pre-Processing-----------------

dataPath = glob.glob("C:/Users/Ali Raza/Desktop/W/*.png")
dataMatrix = np.empty((0,3))

for image in dataPath:
    img = cv2.imread(image)
    shape = img.shape
    img = np.reshape(img,(shape[0]*shape[1],3))
    dataMatrix = np.concatenate((dataMatrix,img))

dataMatrixShape = dataMatrix.shape
dataMatrixRows = dataMatrixShape[0]
dataMatrixColumns = dataMatrixShape[1]

#print(dataMatrix)



#------------------Driver Code--------------------------

print(kMeansClustering(dataMatrix,3))