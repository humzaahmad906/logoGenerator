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
    
    for i in range(dataMatrix.shape[0]):
        distanceArray = centroidsDistance(dataMatrix[i,:],centroidsArray)
        centroidsArray = updateCentroids(dataMatrix[i,:],distanceArray,centroidsArray)    
        
    return centroidsArray

def rgb2HexConverter(centroidsArray):
    centroidsRGBformat = np.empty((0,), dtype = "U25")
    outputArray = np.empty((10,4),dtype = "U25")
    counter = 0
    for i in range(centroidsArray.shape[0]):
        centroidsRGBformat = np.append(centroidsRGBformat,rgb2hex(centroidsArray[i,0],centroidsArray[i,1],centroidsArray[i,2]))
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

#TechColors
dataPathTech = glob.glob("C:/Users/Ali Raza/Google Drive/Logo Generator Project/DataSets/TECH/*.png")
featureMatrixTech = np.empty((0,3))

for image in dataPathTech:
    img = cv2.imread(image)
    img = np.reshape(img,(img.shape[0]*img.shape[1],3))
    featureMatrixTech = np.concatenate((featureMatrixTech,img))


#FoodColors
dataPathFood = glob.glob("C:/Users/Ali Raza/Google Drive/Logo Generator Project/DataSets/FOOD/*.png")
featureMatrixFood = np.empty((0,3))

for image in dataPathFood:
    img = cv2.imread(image)
    img = np.reshape(img,(img.shape[0]*img.shape[1],3))
    featureMatrixFood = np.concatenate((featureMatrixFood,img))

#Metallic Colors
dataPathMetallic = glob.glob("C:/Users/Ali Raza/Google Drive/Logo Generator Project/DataSets/METALLIC/*.png")
featureMatrixMetallic = np.empty((0,3))

for image in dataPathMetallic:
    img = cv2.imread(image)
    img = np.reshape(img,(img.shape[0]*img.shape[1],3))
    featureMatrixMetallic = np.concatenate((featureMatrixMetallic,img))

#Wood Colors
dataPathWood = glob.glob("C:/Users/Ali Raza/Google Drive/Logo Generator Project/DataSets/WOOD/*.png")
featureMatrixWood = np.empty((0,3))

for image in dataPathWood:
    img = cv2.imread(image)
    img = np.reshape(img,(img.shape[0]*img.shape[1],3))
    featureMatrixWood = np.concatenate((featureMatrixWood,img))


#---------------Driver Code-------------------

clustersFood = kMeansClustering(featureMatrixFood,40)
clustersFood = np.asarray(clustersFood, dtype=int)

clustersTech = kMeansClustering(featureMatrixTech,40)
clustersTech = np.asarray(clustersTech, dtype=int)

clustersWood = kMeansClustering(featureMatrixWood,40)
clustersWood = np.asarray(clustersWood, dtype=int)

clustersMetallic = kMeansClustering(featureMatrixMetallic,40)
clustersMetallic = np.asarray(clustersMetallic, dtype=int)



outputDictionary = {"food":rgb2HexConverter(clustersFood), "tech":rgb2HexConverter(clustersTech), "metallic":rgb2HexConverter(clustersMetallic), "wood":rgb2HexConverter(clustersWood)}
