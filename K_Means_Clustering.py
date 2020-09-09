import numpy as np
import cv2
import glob
import random
import math
import matplotlib.pyplot as plt
from colormap import rgb2hex

#-------functions-------------
def centroidsDistance(dataMatrixRow,updatedCentroidsArray):
    distances = np.empty((0,))
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
        centroidsArray[i] = dataMatrix[random.randint(0,dataMatrix.shape[0])]
    #print(centroidsArray)
    for i in range(dataMatrix.shape[0]):
        distanceArray = centroidsDistance(dataMatrix[i,:],centroidsArray)
        centroidsArray = updateCentroids(dataMatrix[i,:],distanceArray,centroidsArray)    
        #print(i)
    return centroidsArray

def rgb2HexConverter(centroidsArray):
    centroidsRGBformat = np.empty((0,), dtype = "U25")
    outputArray = np.empty((10,4),dtype = "U25")
    counter = 0
    for i in range(centroidsArray.shape[0]):
        centroidsRGBformat = np.append(centroidsRGBformat,rgb2hex(centroidsArray[i,0],centroidsArray[i,1],centroidsArray[i,2]))
    #print(centroidsRGBformat[1])
    for i in range(outputArray.shape[0]):
        outputArray[i,0] = centroidsRGBformat[counter]
        counter += 1
        outputArray[i,1] = centroidsRGBformat[counter]
        counter += 1
        outputArray[i,2] = centroidsRGBformat[counter]
        counter += 1
        outputArray[i,3] = centroidsRGBformat[counter]
        counter += 1
  
    return outputArray


#-------------Data Pre-Processing-----------------

dataPath = glob.glob("C:/Users/Ali Raza/Desktop/W/*.png")
featureMatrix = np.empty((0,3))

for image in dataPath:
    img = cv2.imread(image)
    shape = img.shape
    img = np.reshape(img,(shape[0]*shape[1],3))
    featureMatrix = np.concatenate((featureMatrix,img))


#print(dataMatrixShape)

clusters = kMeansClustering(featureMatrix,40)
clusters = np.asarray(clusters,dtype=int)

outputDictionary = {"food":rgb2HexConverter(clusters)}

print(outputDictionary["food"])