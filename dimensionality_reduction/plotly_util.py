from plotly.offline import iplot
import plotly.graph_objs as go


default_marker = go.Marker(size=3, color="#006d2c")


def scatter_matrix(M, marker=default_marker,
                   width=600, height=600,
                   title=None, x_label=None, y_label=None,
                   dims=[0,1]):
    data = [go.Scatter(
        x=M[:, dims[0]],
        y=M[:, dims[1]],
        mode='markers',
        marker=marker
    )]

    layout = go.Layout(
        width=width,
        height=height,
        title=title,
        xaxis=dict(
            title=x_label
        ),
        yaxis=dict(
            title=y_label
        )
    )

    return iplot(go.Figure(data=data, layout=layout))


def scatter_matrix_3d(M, marker=default_marker,
                      width=600, height=600,
                      title=None, x_label=None, y_label=None, z_label=None):
    data = [go.Scatter3d(
        x=M[:, 0],
        y=M[:, 1],
        z=M[:, 2],
        mode='markers',
        marker=marker
    )]

    layout = go.Layout(
        width=width,
        height=height,
        title=title,
        scene=go.Scene(
            xaxis=go.XAxis(title=x_label),
            yaxis=go.YAxis(title=y_label),
            zaxis=go.ZAxis(title=z_label),
        )
    )

    return iplot(go.Figure(data=data, layout=layout))
