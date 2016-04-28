from __future__ import print_function
import numpy as np
import sklearn.decomposition
import sklearn.cluster
import sklearn.mixture
import sklearn.metrics

import plotly
import plotly.tools as pyt
import plotly.plotly as py
import plotly.graph_objs as go

import os

def load_exprmat(exprmat_fn):
  rows = []
  rownames = []

  with open(exprmat_fn) as exprmat:
    colnames = next(exprmat).strip().split(',')
    for row in exprmat:
      fields = row.split(',')
      rownames.append(fields[0])
      rows.append([float(f) for f in fields[1:]])

  data = np.array(rows)
  print(data.shape)
  return (data, colnames, rownames)

def plot(cluster_alg, points, clusters, timepoints, labels):
  colour_scale = 'Viridis'
  scatter1 = go.Scatter(
    x = points[:,0],
    y = points[:,1],
    mode = 'markers',
    marker = {
      'color': clusters,
      'colorscale': colour_scale,
    },
    text = labels
  )
  scatter2 = go.Scatter(
    x = points[:,0],
    y = points[:,1],
    mode = 'markers',
    marker = {
      'color': timepoints,
      'colorscale': colour_scale,
    },
    text = labels
  )
  data = go.Data([scatter1, scatter2])

  fig = pyt.make_subplots(rows=1, cols=2, subplot_titles=('Clusters', 'Timepoints'), print_grid=False)
  fig.append_trace(scatter1, 1, 1)
  fig.append_trace(scatter2, 1, 2)
  n_clusters = len(set(clusters))
  fig['layout'].update(title='Comparing clusering via %s to ground truth (%s clusters)' % (cluster_alg, n_clusters), showlegend=False)
  for idx in range(1, 3):
    fig['layout']['xaxis%s' % idx].update(title='Component 1')
    fig['layout']['yaxis%s' % idx].update(title='Component 2')

  layout = go.Layout(
    hovermode = 'closest',
    xaxis = {
      'title': 'Component 1',
      'domain': [0, 0.45],
    },
    yaxis = {
      'title': 'Component 2',
    },
    xaxis2 = {
      'domain': [0.55, 1],
    },
    yaxis2 = {
      'anchor': 'x2',
    }
  )

  #plotly.offline.plot(fig, filename='%s.html' % cluster_alg.lower())
  plotly.offline.init_notebook_mode()
  plotly.offline.iplot(fig)

def cluster_kmeans(points):
  preds = sklearn.cluster.KMeans(n_clusters=4).fit_predict(points)
  return preds

def cluster_gmm(points):
  preds = sklearn.mixture.GMM(n_components=4).fit_predict(points)
  return preds

def cluster_dpgmm(points):
  preds = sklearn.mixture.DPGMM(n_components=100, alpha=0.5).fit_predict(points)
  return preds

def eval_clustering(truth, clustering):
  score_funcs = [
    sklearn.metrics.adjusted_rand_score,
    sklearn.metrics.v_measure_score,
    sklearn.metrics.adjusted_mutual_info_score,
    sklearn.metrics.mutual_info_score,
  ]
  for score_func in score_funcs:
    print(score_func.__name__, score_func(truth, clustering), sep='\t')

def reduce_dimensionality(exprmat):
  pca = sklearn.decomposition.PCA(n_components=2)
  projected = pca.fit(exprmat).transform(exprmat)
  print('Explained variance ratio: %s' % pca.explained_variance_ratio_)
  return projected

def get_timepoints(samples):
  timepoints = [samp.split('_')[0] for samp in samples]
  keys, values = sorted(set(timepoints), key=lambda tp: int(tp[1:])), range(len(set(timepoints)))
  timepoint_map = dict(zip(keys, values))
  timepoints = [timepoint_map[tp] for tp in timepoints]
  return timepoints

def cluster_and_plot(points, truth, labels, cluster_alg):
  projected = reduce_dimensionality(points)
  clusters = cluster_alg(points)
  eval_clustering(truth, clusters)
  plot(cluster_alg.__name__, projected, clusters, truth, labels)

def main():
  exprmat, genes, samples = load_exprmat(os.path.dirname(__file__) + '../data/expression_matrix.csv')
  timepoints = get_timepoints(samples)

  for cluster_alg in (cluster_kmeans, cluster_gmm, cluster_dpgmm):
    print(cluster_alg.__name__)
    cluster_and_plot(exprmat, timepoints, samples, cluster_alg)

if __name__ == '__main__':
  main()
