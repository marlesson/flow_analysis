from argparse import ArgumentParser
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from util_arrows import *

def generate_sum_vectors(df, labels, directory):
    clusters = {}
    for index, value in enumerate(labels):
        if value == -1:
            continue
        if value not in clusters:
            clusters[value] = []
        clusters[value].append(index)
    
    shape = (df.x1.max(), df.y1.max())
    
    clusters_sum = []
    for key, value in clusters.items():
        df_cluster = df.filter(clusters[key], axis=0)
        df_cluster.to_csv('{}/arrows_cluster{}.csv'.format(directory,key))

        df_cluster_mean = df_cluster.mean()
        df_cluster_sum = df_cluster.agg('sum')
        df_cluster_mean['x'] = df_cluster_sum['x']
        df_cluster_mean['y'] = df_cluster_sum['y']
        clusters_sum.append(df_cluster_mean)

    print(pd.DataFrame(clusters_sum))
    fig = plot_arrows(pd.DataFrame(clusters_sum), shape)
    fig.savefig("{}/clusters_sum.png".format(directory))

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument("-f", "--file", dest="filename",
                      help="CSV with vectors.")

  parser.add_argument("-d", "--max_distance", dest="max_distance",
                      help="The maximum distance between two samples for them to be considered as in the same neighborhood.",
                      type=float, default=0.5)

  parser.add_argument("-m", "--min_samples", dest="min_samples",
                      help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.",
                      type=int, default=5)

  args = parser.parse_args()
  print(args)

  directory = "/".join(args.filename.split("/")[:-1]) 

  df = pd.read_csv(args.filename)
  df_norm = (df - df.mean()) / (df.max() - df.min())
  matrix = df_norm.drop(columns=["x","y"]).values[:,1:]
  db = DBSCAN(eps=args.max_distance, min_samples=args.min_samples).fit(matrix)
  core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  print(db.labels_)

  n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
  print("Clusters: ", n_clusters_)
  
  shape = (df.x1.max(), df.y1.max())
  fig   = plot_arrows(df, shape, getColors(db.labels_))
  fig.savefig("{}/clusters.png".format(directory))
  generate_sum_vectors(df, db.labels_, directory)
