from __future__ import print_function
import rpy2.robjects as robjects
import numpy as np
import sklearn.decomposition
import sklearn.cluster
import sklearn.mixture

import plotly
import plotly.tools as pyt
import plotly.plotly as py
import plotly.graph_objs as go

def load_exprmat():
  robjects.r['load']('ExpressionMatrix.rda')
  exprmat = robjects.r('ExpressionMatrix')
  colnames = list(exprmat.colnames)
  rownames = list(exprmat.rownames)
  return (np.array(exprmat), colnames, rownames)

def plot(cluster_alg, points, clusters, timepoints, labels):
  scatter1 = go.Scatter(
    x = points[:,0],
    y = points[:,1],
    mode = 'markers',
    marker = {
      'color': clusters,
      'colorscale': 'Viridis',
    },
    text = labels
  )
  scatter2 = go.Scatter(
    x = points[:,0],
    y = points[:,1],
    mode = 'markers',
    marker = {
      'color': timepoints,
      'colorscale': 'Viridis',
    },
    text = labels
  )
  data = go.Data([scatter1, scatter2])

  fig = pyt.make_subplots(rows=1, cols=2, subplot_titles=('Clusters', 'Timepoints'))
  fig.append_trace(scatter1, 1, 1)
  fig.append_trace(scatter2, 1, 2)
  fig['layout'].update(title='Comparing clusering via %s to ground truth' % cluster_alg, showlegend=False)
  for idx in range(1, 3):
    fig['layout']['xaxis%s' % idx].update(title='Component 1')
    fig['layout']['yaxis%s' % idx].update(title='Component 2')

  layout = go.Layout(
    title = 'Gene expression levels',
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

  #figure = go.Figure(data=data, layout=layout)
  plotly.offline.plot(fig, filename='%s.html' % cluster_alg.lower())

def cluster_kmeans(points):
  preds = sklearn.cluster.KMeans(n_clusters=4).fit_predict(points)
  return preds

def cluster_gmm(points):
  preds = sklearn.mixture.GMM(n_components=4).fit_predict(points)
  return preds

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

def main():
  exprmat, samples, genes = load_exprmat()
  # Add 1 to avoid taking log(0).
  exprmat = np.log(exprmat + 1)
  exprmat = exprmat.T

  projected = reduce_dimensionality(exprmat)
  timepoints = get_timepoints(samples)

  clusters_kmeans = cluster_kmeans(projected)
  clusters_gmm = cluster_gmm(projected)

  plot('kmeans', projected, clusters_kmeans, timepoints, samples)
  plot('GMM', projected, clusters_gmm, timepoints, samples)

main()
