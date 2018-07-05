from argparse import ArgumentParser
import numpy as np
import pandas as pd
import math
import sys
import os
import matplotlib.pyplot as plt
#from util import *
from util_arrows import *

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument("-f", "--file", dest="filename",
                      help="File")

  parser.add_argument("-out", "--output", dest="output",
                      help="OutputPath", default='output')

  parser.add_argument("-k", "--kernel", dest="kernel",
                      help="Deep Kernel",
                      type=int, default=10)

  parser.add_argument("-p", "--pass", dest="pass_frame",
                      help="Frames",
                      type=int, default=10)

  args = parser.parse_args()
  print(args)
  
  # Leitura do dataset
  df = pd.read_csv(args.filename)
  df = df[df['size'] > 0]

  shape = (df.x2.max(), df.y2.max())
  frame_max = df.frame.max()

  # Cria pasta de output
  filename         = args.filename.split("/")[-1].split(".")[0]
  directory        = args.output + "/" + filename 
  base_directory   = directory+"/"+"base_vectors"
  filter_directory = directory+"/"+"filter_vectors"
  if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(base_directory)
    os.makedirs(filter_directory)


  # Orientação por Angulo (agrupado)
  df_angle = df.groupby(['angle']).agg({'size': 'sum'}).reset_index()
  fig, ax = compass(df_angle.angle.values, df_angle['size'].values)
  fig.savefig(directory+'/plot.png')

  # Plot per Frame
  for frame in range(1, frame_max, args.pass_frame):
    print(frame, "/", frame_max)
    # Filter per frame
    df_filter = df[df.frame <= frame]

    if len(df_filter) > 0:
      # Plot Base Arrows
      fig       = plot_arrows(df_filter, shape)
      fig.savefig("{}/{}.png".format(base_directory, frame))

      # Block Partition
      df_pooling = block_partition(df_filter, shape, args.kernel)
      fig        = plot_arrows(df_pooling, shape)
      fig.savefig("{}/{}.png".format(filter_directory, frame))
