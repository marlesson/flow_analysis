import numpy as np
import cv2
import time
from scipy import signal

# def draw_flow(img, flow, step=16):
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
#     fx, fy = flow[y,x].T
#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
#     cv2.polylines(vis, lines, 0, (0, 255, 0))
#     for (x1, y1), (x2, y2) in lines:      
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import cv2 as cv
import math
import matplotlib.cm as cm
from scipy.signal import convolve2d
import random


def Lucas_Kanade(I1, I2, features, k = 1):
    #oldframe = cv.imread(image1)
    #I1 = cv.cvtColor(oldframe, cv.COLOR_BGR2GRAY)

    #newframe = I2#cv.imread(image2)
    #I2 = cv.cvtColor(newframe, cv.COLOR_BGR2GRAY)

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

if __name__ == '__main__':
    import sys
    color = np.random.randint(0, 255, (500, 3))

    # parameter to get features
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)    

    cap   = cv2.VideoCapture('data/sample.flv')
    #cap   = cv2.VideoCapture('data/slow.flv')

    #cap = cv2.VideoCapture(0)
    #cap.set(3,1280)
    #cap.set(4,1024)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray       = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    #print(len(p0))
    p0             = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)    
    while True:
        
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        # Optical Flow
        p1, st = Lucas_Kanade(old_gray, frame_gray, p0, 7)
    
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i, (new,old) in enumerate(zip(good_new, good_old)):
            a,b   = new.ravel()
            c,d   = old.ravel()
            #t = 2
            #if np.fabs(a - c) < t or np.fabs(b - d) < t:
            #    continue

            mask  = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5, color[i].tolist(),-1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame',img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0       = good_new.reshape(-1,1,2)
        print(len(p0))

    cv2.destroyAllWindows()
    cap.release()