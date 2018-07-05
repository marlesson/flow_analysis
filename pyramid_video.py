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
from skimage.transform import pyramid_gaussian

from util import *

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument("-f", "--file", dest="filename",
                      help="Video", metavar="FILE")

  parser.add_argument("-out", "--out", dest="out",
                      help="Video", metavar="FILE", default='outvideo')

  parser.add_argument("-max_layer", "--max_layer",
                      dest="max_layer", type=int, default=0)

  args = parser.parse_args()
  print(args)

  downscale = 2

  # VideoCapture and Save
  cap   = cv2.VideoCapture(args.filename)

  # Define the codec and create VideoWriter object
  # Pyramid Gaussian
  
  outpy  = {}
  w, h   = (int(cap.get(3)), int(cap.get(4)))
  for py in range(args.max_layer):
    fourcc    = cv2.VideoWriter_fourcc('F','M','P','4')
    outpy[py] = cv2.VideoWriter(args.out+"_"+str(py)+".avi", fourcc, 20.0, (int(w), int(h)))
    w = w/downscale
    h = h/downscale
  
  # per Frame
  while True:
      # readframe
      ret, frame = cap.read()
      if not ret:
          break
      # Show imagem
      #cv2.imshow('frame',frame)

      # Pyramid Gaussian
      pyramid = tuple(pyramid_gaussian(frame, downscale=downscale,  max_layer=args.max_layer))
      for py in range(args.max_layer):
        outpy[py].write(np.uint8(pyramid[py]*255))

      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
  
  for py in range(args.max_layer):
    outpy[py].release()

  cap.release()
  cv2.destroyAllWindows()
