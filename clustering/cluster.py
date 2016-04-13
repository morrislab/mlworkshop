import rpy2.robjects as robjects
import numpy as np
import sklearn.decomposition
import sklearn.cluster

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def load_exprmat():
  robjects.r['load']('ExpressionMatrix.rda')
  exprmat = robjects.r('ExpressionMatrix')
  colnames = list(exprmat.colnames)
  rownames = list(exprmat.rownames)
  return (np.array(exprmat), colnames, rownames)

def plot(points, clusters, labels):
  trace = go.Scatter(
    x = points[:,0],
    y = points[:,1],
    mode = 'markers',
    marker = {
      'color': clusters,
      'colorscale': 'Viridis',
    },
    text = labels
  )
  layout = go.Layout(
    title = 'Gene expression levels',
    hovermode = 'closest',
    xaxis = {
      'title': 'Component 1',
    },
    yaxis = {
      'title': 'Component 2',
    }
  )

  data = go.Data([trace])
  figure = go.Figure(data=data, layout=layout)
  plotly.offline.plot(figure, filename='genes.html')

def cluster(points):
  preds = sklearn.cluster.KMeans(n_clusters=4).fit_predict(points)
  return preds

def main():
  exprmat, colnames, rownames = load_exprmat()
  # Add 1 to avoid taking log(0).
  exprmat = np.log(exprmat + 1)

  pca = sklearn.decomposition.PCA(n_components=2)
  projected = pca.fit(exprmat).transform(exprmat)
  print('Explained variance ratio: %s' % pca.explained_variance_ratio_)
  clusters = cluster(exprmat)
  plot(projected, clusters, rownames)

main()
