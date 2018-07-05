from argparse import ArgumentParser
import cv2
import time
from scipy import signal
import numpy as np
import cv2 as cv
import math
import random
import sys
import matplotlib.pyplot as plt
from util import *

perspective_samples = {
    "original": np.array([[498, 582], [545, 580], [534, 258], [566, 258]]), 
    "flat": np.array([[500, 580], [530, 580], [500, 300], [530, 300]])
}

remove_perspective_matrix = calculate_perspective_transformation(perspective_samples['flat'], 
    perspective_samples['original'], 0.7, [200,100])

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument("-f", "--file", dest="filename",
                      help="Video", metavar="FILE")

  parser.add_argument("-out", "--out", dest="out",
                      help="Video", metavar="FILE", default='outvideo')

  parser.add_argument("-c", "--maxCorners",
                      dest="maxCorners", type=int, default=100)

  parser.add_argument("-s", "--sizeWinlk",
                      dest="sizeWinlk", type=int, default=10)

  parser.add_argument("-g", "--groupAngle",
                      dest="groupAngle", action='store_true')

  args = parser.parse_args()
  print(args)

  # VideoCapture and Save
  cap   = cv2.VideoCapture(args.filename)

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
  out    = cv2.VideoWriter(args.out+".avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


  # Take first frame and find corners in it
  plt.figure(1)
  ret, old_frame = cap.read()
  old_gray       = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
  plt.subplot(121)
  plt.imshow(old_gray, cmap="gray")
  plt.scatter(perspective_samples["original"][:,0],perspective_samples["original"][:,1], marker=".")
  old_gray2      = transform_image(old_gray, remove_perspective_matrix)
  plt.subplot(122)
  plt.imshow(old_gray2, cmap="gray")
  plt.scatter(perspective_samples["flat"][:,0],perspective_samples["flat"][:,1], marker=".")
  plt.show()
  # parameter to get features track
  feature_params = dict(maxCorners=args.maxCorners,
                        qualityLevel=0.3,
                        minDistance=7,
                        blockSize=7)   
  p0             = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
  
  # Create a mask image for drawing purposes
  mask    = np.zeros_like(old_frame)    
  i_frame = 1

  # Log pixels flow
  log_flow = []

  while True:
      # readframe
      ret, frame = cap.read()

      if not ret:
          break

      # Resent FeaturesTrack
      if i_frame % 5 == 0:
          p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
      
      #print(len(p0))

      # Frame in Grayscale
      frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      #frame_gray = transform_image(frame_gray, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    

      # Optical Flow
      p1, st = Lucas_Kanade(old_gray, frame_gray, p0, args.sizeWinlk)
  
      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]

      # draw the tracks
      for i, (new,old) in enumerate(zip(good_new, good_old)):
          a,b   = new.ravel()
          c,d   = old.ravel()

          x     = a-c
          y     = b-d
          angle = math.atan2(y, x)
          size  = math.sqrt(x**2+y**2)
          
          log_flow.append([i_frame, i, int(a), int(b), int(c), 
                            int(d), int(x), int(y), angle, size])
          #print((c,d), " -> ", (a,b), " ", (x, y), " ", angle, " ", size)
          # continue if diff between pixels positions is less 1
          t = 1
          if np.fabs(x) <= t and np.fabs(y) <= t:
              continue
          
          mask  = cv2.line(mask, (a,b), (c,d), rad2rgb(angle, cluster=args.groupAngle), 2)
          frame = cv2.circle(frame,(a,b), 3, rad2rgb(angle, cluster=args.groupAngle), -1)

      img = cv2.add(frame, mask)
      
      # Show imagem
      cv2.imshow('frame',img)

      # write the flipped frame
      out.write(img)

      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break

      # Now update the previous frame and previous points
      old_gray = frame_gray.copy()
      p0       = good_new.reshape(-1, 1, 2)
      i_frame  = i_frame+1

  #Save Log flow
  save_logflow(log_flow, args.out+".csv")

  out.release()
  cap.release()
  cv2.destroyAllWindows()
