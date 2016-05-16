import numpy as np
from plotly.offline import iplot
from plotly import graph_objs as go


def cost_function(X, Y, theta):
    return np.sum(np.power(np.dot(X, theta) - Y, 2)) / (2 * len(X))


def gradient_descent(X, Y, theta, alpha=0.02):
    return theta - ((alpha/len(X)) * np.dot(X.T, np.dot(X, theta) - Y))


def calculate_cost_grid(X, Y, low, high, size):

    grid = np.zeros((size, size))
    lsp = np.linspace(low, high, size)

    for i, x in enumerate(lsp):
        for j, y in enumerate(lsp):
            grid[i, j] = cost_function(X, Y, np.array([x, y]))

    return grid


def generate_line(slope, intercept, n=50):
    theta = [intercept, slope]
    noise = np.random.normal(scale = 2, size = n)
    x = np.linspace(-10, 10, n)
    X = np.column_stack((np.ones(n), x))
    Y = (theta[1] * (x + noise)) + theta[0]
    return X, Y


def plot_cost_and_trace(X, Y, trace, low=-10, high=10, size=100):
    gr = calculate_cost_grid(X, Y, low, high, size)
    lsp = np.linspace(low, high, size)

    contour = go.Contour(z = np.log(gr), x = lsp, y = lsp, colorscale='Jet', contours={'coloring': 'heatmap'})

    descent = go.Scatter(x = trace[:, 1], y = trace[:, 0], mode='markers')

    iplot(go.Figure(data=[contour, descent], layout=go.Layout(hovermode='closest', width=600, height=600)))
