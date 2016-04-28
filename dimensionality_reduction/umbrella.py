# Umbrella example
import numpy as np
import numpy.random as random
import plotly.graph_objs as go
from plotly_util import scatter_matrix_3d


class Umbrella:
    def __sample_circle(self, n):
        t = 2 * np.pi * random.uniform(size=n)
        u = random.uniform(size=n) + random.uniform(size=n)
        r = np.vectorize(lambda u: 2 - u if u > 1 else u)(u)
        return np.column_stack((r * np.cos(t), r * np.sin(t)))

    def __init__(self, n, handle_thinner=10):
        canopy = np.column_stack((self.__sample_circle(n), np.repeat(0, n)))
        handle = np.column_stack((np.repeat(0, n // handle_thinner),
                                  np.repeat(0, n // handle_thinner),
                                  -random.uniform(high=2,
                                                  size=n // handle_thinner)))
        self.matrix = np.row_stack((canopy, handle))
        self.color = np.apply_along_axis(
            lambda row: "#d03" if row[2] == 0 else "#444", 1, self.matrix)
        self.marker = go.Marker(
            size=4,
            color=self.color
        )

    def plot(self):
        scatter_matrix_3d(self.matrix, marker=self.marker)


def test_umbrealla():
    umbrella = Umbrella(100)
    assert umbrella.matrix.shape == (110, 3)
