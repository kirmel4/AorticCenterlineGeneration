import numpy as np

from aaa.geometry.constants import ZNORMAL

class TreeVertex(object):
    def __init__(self, point):
        self._point = point

        self._distance = 0.
        self._unit_vector = ZNORMAL.copy()

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, args):
        raise AttributeError()

    @property
    def normal(self):
        return self._unit_vector / np.linalg.norm(self._unit_vector)

    @normal.setter
    def normal(self, args):
        raise AttributeError()

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, args):
        raise AttributeError()

class TreeBiffurcation(object):
    def __init__( self,
                  source_vertex_idx,
                  biffurcation_vertex_idx,
                  sink_vertex_ids ):
        self._source_vertex_idx = source_vertex_idx
        self._biffurcation_vertex_idx = biffurcation_vertex_idx
        self._sink_vertex_ids = sink_vertex_ids

        assert len(self._sink_vertex_ids) == 2
