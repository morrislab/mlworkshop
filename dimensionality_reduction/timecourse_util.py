import plotly.graph_objs as go


def timecourse_point(x):
    return int(x[1:x.find("_")])


def timecourse_marker(data):
    return go.Marker(
        size=8,
        line=dict(
            width=2
        ),
        color=data.index.map(timecourse_point),
        colorscale=[[0, 'rgb(255,204,92)'], [1, 'rgb(189,0,38)']],
        showscale=True
    )


def test_timecourse_point():
    assert timecourse_point("T24_CT_A04") == 24
