from skimage.io import imread, imshow
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as pl
import numpy as np
from scipy import ndimage
import cv2 as cv
from keras.datasets import mnist
from skimage.color import rgb2gray
import csv
import pandas
from numpy import linalg as LA
from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


histo_arr = [0,0,0,0,0,0,0,0,0]
def calac_gradiants(img):
    dx = ndimage.sobel(img, 0)
    dy = ndimage.sobel(img, 1) 
    mag = np.hypot(dx, dy)
    dx=dx+0.0001
    theta=np.arctan(dy/dx)
    theta=np.rad2deg(theta)
    theta=theta%180
    return mag,theta

def fill_in():
    temp=np.zeros((9))
    for i in range (9):
        temp[i]=histo_arr[i]            
    return temp


def divde_blocks(mag,theta):
    result=list()
    x=0
    y=0
    for i in range(0,32,8): 
        for j in range(0,32,8):
            theat8=theta[i:i+8,j:j+8]
            mag8=mag[i:i+8,j:j+8]
            calac_histogram(mag8 , theat8)
            temp=fill_in()
            result.append(temp)
        x=x+1
        y=y+1
        histo_arr=[0,0,0,0,0,0,0,0,0]
    return result

def check_theta (angle , hist_bins ):
    
    for i in range (8):
        if (angle >= hist_bins[i] and angle < hist_bins[i+1] ):   
            return hist_bins[i] , hist_bins[i+1]
        
    

    
def set_histogram (lower , upper , angle , mag1):
    var1 = (abs( lower - angle )) / 20
    var2 = (abs( upper - angle )) / 20
    min1 =  min (var1 , var2) * mag1
    max1 =  max (var1, var2)  * mag1
        
    if (var1 == (min1/mag1)):    
        index = hist_bins.index(lower)
        histo_arr[index]+=max1
        index = hist_bins.index(upper)
        histo_arr[index]+=min1
        
    else:                 
        index = hist_bins.index(upper)
        histo_arr[index]+=max1
        index = hist_bins.index(lower)
        histo_arr[index]+=min1
 

     
def calac_histogram(mag8,theat8):
    for i in range (0 , 8):
        for j in range (0 , 8):
            if(theat8[i][j]>=170):
                index = hist_bins.index(170)
                histo_arr[index]+=(theat8[i][j]*mag8[i][j])
            elif(theat8[i][j]<10):
                index = hist_bins.index(10)
                histo_arr[index]+=(theat8[i][j]*mag8[i][j])
                
            else:        
                lower , upper = check_theta(theat8[i][j],hist_bins)
                set_histogram(lower,upper,theat8[i][j],mag8[i][j])
        

hist_bins = [10,30,50,70,90,110,130,150,170]
histo180 = np.array([20,40,60,80,100,120,140,160,180])

arr = np.array([[2	,2	,3	,3], [1,5,6,2], [2,6,5,3],[1,1,2,1]])

def get_bolock_sum(mat):
    sum_arr=list()
    for i in range (0,36,4):
        sum1=0
        for j in range(i,i+4):
            for x in range(9):
                sum1=sum1+mat[j][x]

        sum_arr.append(sum1)
    return sum_arr
def normlaition(lst,sum_arr):
    n=0
    temp=list()
    for i in range(0,36,4):
        for j in range(i,i+4):
            for x in range(9):
                temp.append(lst[j][x]/sum_arr[n])
        n=n+1
    return temp


def duplicate(mat):
    lst=list()
    sum_arr=list()
    sum=0
    x=0
    for i in range(3):
        for j in range (3):
            
            lst.append(mat[i][j])
            lst.append(mat[i][j+1])
            lst.append(mat[i+1][j])
            lst.append(mat[i+1][j+1])
                
    return lst
   

def hog(img):
        
    mag,theta=calac_gradiants(img)
    mat=(divde_blocks(mag,theta))
    b=np.array(mat)
    b=b.reshape(4,4,9)    
    lst=duplicate(b)
    lst=np.array(lst)
    sum1=get_bolock_sum(lst)
    final=normlaition(lst,sum1)
    return final    

(train_X, train_y), (test_X, test_y) = mnist.load_data()
feature_vec=list()
featureVectorDataFrame=pandas.DataFrame()
testvec=pandas.DataFrame()
for i in range(10):
        img=train_X[i]
        img=cv.resize(img,(32,32))
        res=hog(img)
        feature_vec.append(res)
        
  

model = LinearSVC()

model.fit(feature_vec, train_y[0:10])

# Evaluate the classifier
print(" Evaluating classifier on test data ...")
predictions = model.predict(feature_vec)

print("\nAccuracy Score : ",accuracy_score(test_y[0:10], predictions), "\n")
#joblib.dump(model, "C:/Users/Mohamed/Desktop/hog_v1.npy")
