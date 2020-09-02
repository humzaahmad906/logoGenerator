import numpy as np
import cv2
import glob
import random
import math
import matplotlib.pyplot as plt


#DATA-READ and Matrix update-----------
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

#-----------K-MEANS-CLUSTERING------------
#function for taking K(clusters) values

def K_Means_Clustering(matrix, K_input):

    matrixShape = matrix.shape
    matrixRows = matrixShape[0]
    K_array = np.empty((K_input,matrixShape[1])) #Initialized numpy array for storing random and updated values of K
    Ks = np.empty((K_input)) #Numpy array containing values of K from eulidean distance calculations (for checking minimum K)
  
    
    #selecting random points for no. of clusters
    for i in range (K_input):
        K_array[i] = matrix[random.randint(0,matrixRows)]
    
    #iterating each row of feature matrix
    for i in range (matrixRows):
        feature_row = matrix[i]
        
        #checking nearest K
        for j in range(len(K_array)):  
            subDist = 0.00
            for n in range(matrixShape[1]):
                subDist += float((feature_row[n] - K_array[j,n])**2)
                Ks[j] = subDist

        K = Ks.tolist()
        minK = K.index(min(K)) #Gives index of cluster with minimum value(distance)

        #updating Ks
        K_array[minK,0] = (feature_row[0]+K_array[minK,0])/2
        K_array[minK,1] = (feature_row[1]+K_array[minK,1])/2
        K_array[minK,2] = (feature_row[2]+K_array[minK,2])/2
    
    return K_array

K_array = K_Means_Clustering(dataMatrix,3)  #input(feature matrix, no of clusters needed)
print(K_array)