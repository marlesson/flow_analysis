import time
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
from scipy.signal import convolve2d
import random
import sys
from sklearn.externals import joblib
from sklearn.cluster import KMeans

def save_logflow(log, file="log_flow.csv"):
  '''
  Save log file
  '''
  log_str = [", ".join(str(l) for l in line) for line in log]

  with open(file, 'w') as f:
    f.write("frame,i,x1,y1,x2,y2,x,y,angle,size\n")
    for line in log_str:
      f.write(line+"\n")


def Lucas_Kanade(I1, I2, features, k = 1):
    Gx  = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy  = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image


    Ix  = (convolve2d(I1, Gx) + convolve2d(I2, Gx)) / 2 #smoothing in x direction

    Iy  = (convolve2d(I1, Gy) + convolve2d(I2, Gy)) / 2 #smoothing in y direction
    It1 = convolve2d(I1, Gt1) + convolve2d(I2, Gt2)   #taking difference of two images using gaussian mask of all -1 and all 1

    #features = cv.goodFeaturesToTrack(I1, mask = None, **feature_params)  #using opencv function to get feature for which we are plotting flow
    feature  = np.int32(features)
    feature  = np.reshape(feature, newshape=[-1, 2])

    u = np.ones(Ix.shape)
    v = np.ones(Ix.shape)
    status=np.zeros(feature.shape[0]) # this will tell change in x,y
    A = np.zeros((2, 2))
    B = np.zeros((2, 1))
    mask = np.zeros_like(I1)

    newFeature=np.zeros_like(feature)

    max_x, max_y = I1.shape
    """Assumption is  that all the neighbouring pixels will have similar motion. 
    Lucas-Kanade method takes a 3x3 patch around the point. So all the 9 points have the same motion.
    We can find (fx,fy,ft) for these 9 points. So now our problem becomes solving 9 equations with two unknown variables which is over-determined. 
    A better solution is obtained with least square fit method.
    Below is the final solution which is two equation-two unknown problem and solve to get the solution.
                               U=Ainverse*B 
    where U is matrix of 1 by 2 and contains change in x and y direction(x==U[0] and y==U[1])
    we first calculate A matrix which is 2 by 2 matrix of [[fx**2, fx*fy],[ fx*fy fy**2] and now take inverse of it
    and B is -[[fx*ft1],[fy,ft2]]"""
    for a, i in enumerate(feature):
        x, y = i

        A[0, 0] = np.sum((Ix[y-k:y+1+k, x-k:x+1+k]) ** 2)
        A[1, 1] = np.sum((Iy[y-k:y+1+k, x-k:x+1+k]) ** 2)

        A[0, 1] = np.sum(Ix[y-k:y+1+k, x-k:x+1+k] * Iy[y-k:y+1+k, x-k:x+1+k])
        A[1, 0] = np.sum(Ix[y-k:y+1+k, x-k:x+1+k] * Iy[y-k:y+1+k, x-k:x+1+k])

        Ainv = np.linalg.pinv(A)

        B[0, 0] = -np.sum(Ix[y-k:y+1+k, x-k:x+1+k] * It1[y-k:y+1+k, x-k:x+1+k])
        B[1, 0] = -np.sum(Iy[y-k:y+1+k, x-k:x+1+k] * It1[y-k:y+1+k, x-k:x+1+k])
        prod    = np.matmul(Ainv, B)
        
        u[y, x] = prod[0]
        v[y, x] = prod[1]

        newFeature[a]=[np.min([max_y-k*2+1, np.int32(x-u[y,x])]), 
                        np.min([max_x-k*2+1, np.int32(y-v[y,x])])]

        #print([max_x-k*2+1, np.int32(x-u[y,x])])

        #newFeature[a]=[np.int32(x-u[y,x]), np.int32(y-v[y,x])]

        if np.int32(x+u[y,x])==x and np.int32(y+v[y,x])==y:    # this means that there is no change(x+dx==x,y+dy==y) so marking it as 0 else
            status[a]=0
        else:
            status[a]=1 # this tells us that x+dx , y+dy is not equal to x and y

    um = np.flipud(u)
    vm = np.flipud(v)

    return newFeature, status


kmeans = joblib.load('kmeans.pkl') 
def predict_cluster(val):
  return kmeans.predict(val)

def rad2rgb(rad, cluster = False):
    
    if cluster:
      c   = predict_cluster(rad)
      rgb = floatRgb(c, 0, 1)
    else:
      rgb = floatRgb(rad, -math.pi, math.pi)

    return rgb

def floatRgb(mag, cmin, cmax):
       """
       Return a tuple of floats between 0 and 1 for the red, green and
       blue amplitudes.
       """
       try:
              # normalize to [0,1]
              x = float(mag-cmin)/float(cmax-cmin)
       except:
              # cmax = cmin
              x = 0.5
       blue = min((max((4*(0.75-x), 0.)), 1.))*255
       red  = min((max((4*(x-0.25), 0.)), 1.))*255
       green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))*255
       return [int(red), int(green), int(blue)]
