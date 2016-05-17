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
  return (data, colnames, rownames)

def plot_expression(cluster_alg, points, clusters, timepoints, labels):
  colour_scale = 'RdYlBu'
  scatter1 = go.Scatter(
    x = points[:,0],
    y = points[:,1],
    mode = 'markers',
    marker = {
      'color': timepoints,
      'colorscale': colour_scale,
    },
    text = labels
  )
  scatter2 = go.Scatter(
    x = points[:,0],
    y = points[:,1],
    mode = 'markers',
    marker = {
      'color': clusters,
      'colorscale': colour_scale,
    },
    text = labels
  )
  data = go.Data([scatter1, scatter2])

  fig = pyt.make_subplots(rows=1, cols=2, subplot_titles=('Timepoints', 'Clusters'), print_grid=False)
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

def cluster_kmeans(points, nclusters):
  preds = sklearn.cluster.KMeans(n_clusters=nclusters).fit_predict(points)
  return preds

def cluster_gmm(points, nclusters):
  preds = sklearn.mixture.GMM(n_components=nclusters).fit_predict(points)
  return preds

def cluster_agglomerative(points, nclusters):
  preds = sklearn.cluster.AgglomerativeClustering(linkage='ward', n_clusters=nclusters).fit_predict(points)
  return preds

def cluster_dpgmm(points, nclusters):
  # We ignore nclusters here.
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
  #tsne = sklearn.manifold.TSNE(n_components=2)
  projected = pca.fit_transform(exprmat)
  print('Explained variance ratio: %s %s' % (np.sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_))
  return projected

def get_timepoints(samples):
  timepoints = [samp.split('_')[0] for samp in samples]
  #timepoints = [t == 'T0' and 'T0' or 'T1' for t in timepoints]
  #print(timepoints)
  keys, values = sorted(set(timepoints), key=lambda tp: int(tp[1:])), range(len(set(timepoints)))
  timepoint_map = dict(zip(keys, values))
  timepoints = [timepoint_map[tp] for tp in timepoints]
  return timepoints

def cluster_and_plot(points, truth, labels, cluster_alg, nclusters):
  projected = reduce_dimensionality(points)
  clusters = cluster_alg(points, nclusters=nclusters)
  eval_clustering(truth, clusters)
  print('Silhouette', sklearn.metrics.silhouette_score(points, clusters))
  #plot(cluster_alg.__name__, projected, clusters, truth, labels)

def generate_simulated_points():
  num_classes = 3
  points_per_class = 40
  points = np.zeros((num_classes * points_per_class, 2))
  labels = np.zeros(num_classes * points_per_class)
  means = [(-3, -2), (-2, -4), (3, 2)]
  cov = np.eye(2)

  for idx in range(num_classes):
    rows = range(idx * points_per_class, (idx + 1) * points_per_class)
    points[rows,:] = np.random.multivariate_normal(means[idx], cov, points_per_class)
    labels[rows] = idx

  idxs = np.random.permutation(np.arange(num_classes * points_per_class))
  return (points[idxs,:], labels[idxs])

def plot_simulated(points, preds, labels, cluster_alg):
  colour_scale = 'Portland'
  symbols = { 0: 'triangle-up', 1: 'cross', 2: 'star'}

  data = []
  for cls in symbols.keys():
    idxs = np.where(labels == cls)[0]
    scatter = go.Scatter(
      name = 'Class %s' % cls,
      x = points[idxs,0],
      y = points[idxs,1],
      mode = 'markers',
      marker = {
        'color': preds[idxs],
        'symbol': symbols[cls],
        'colorscale': colour_scale,
      }
    )
    data.append(scatter)
  data = go.Data(data)

  layout = go.Layout(
    title = 'Simulated data plotted via %s' % cluster_alg,
    hovermode = 'closest',
    xaxis = {
      'title': 'x',
    },
    yaxis = {
      'title': 'y',
    }
  )

  fig = go.Figure(data=data, layout=layout)
  plotly.offline.init_notebook_mode()
  plotly.offline.iplot(fig)
  #plotly.offline.plot(fig, filename='simulated.html')

def plot_line_chart(xvals, yvals, title, xtitle, ytitle):
  scatter = go.Scatter(
    x = xvals,
    y = yvals,
    mode = 'lines+markers',
  )
  data = go.Data([scatter])

  layout = go.Layout(
    title = title,
    hovermode = 'closest',
    xaxis = {
      'title': xtitle,
    },
    yaxis = {
      'title': ytitle,
    }
  )

  fig = go.Figure(data=data, layout=layout)
  plotly.offline.init_notebook_mode()
  plotly.offline.iplot(fig)

def run_simulated():
  points, labels = generate_simulated_points()
  preds = sklearn.cluster.KMeans(n_clusters=3).fit_predict(points)
  preds = sklearn.mixture.GMM(n_components=3).fit_predict(points)
  plot_simulated(points, preds, labels)

def main():
  run_simulated()
  return

  exprmat, genes, samples = load_exprmat(os.path.dirname(__file__) + '../data/expression_matrix.csv')
  timepoints = get_timepoints(samples)

  for cluster_alg in (cluster_kmeans, cluster_gmm, cluster_dpgmm, cluster_agglomerative):
    print(cluster_alg.__name__)
    cluster_and_plot(exprmat, timepoints, samples, cluster_alg, nclusters=4)
    print()

if __name__ == '__main__':
  main()
