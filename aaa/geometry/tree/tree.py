import numpy as np

from collections import defaultdict
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from aaa.utils import pairwise
from aaa.geometry.tree.tree_errors import TreeConstructionError
from aaa.geometry.tree.tree_structures import TreeVertex, TreeBiffurcation

class Tree(object):
    def __init__(self, points, *, root_idx=0):
        """
            :NOTE:
                assume points in isotropic space, point order is zxy

            :args:
                points (list of (float, float, float)): list of points belonging to the tree
                root_idx (int): index of a point from which the tree begins
        """

        self._vertices = { }
        self._edges = defaultdict(list)

        self._root_vertex_idx = root_idx
        self.__construct_tree(points)

        self._biffurcations = []
        self.__init_tree_attributes()

    def __construct_tree(self, points):
        if len(points) == 0:
            raise TreeConstructionError('point list is empty')
        elif self._root_vertex_idx < 0 or len(points) <= self._root_vertex_idx:
             raise TreeConstructionError('root index is out point range')
        else:
            distances = distance_matrix(points, points)
            graph_matrix = minimum_spanning_tree(distances)

            for idx, _ in enumerate(points):
                self._vertices[idx] = TreeVertex(points[idx])

            for ridx, cidx in zip(*graph_matrix.nonzero()):
                self._edges[ridx].append(cidx)
                self._edges[cidx].append(ridx)

    def get_root_vertex(self):
        return self._vertices[self._root_vertex_idx]

    def get_vertex(self, idx):
        return self._vertices[idx]

    def __init_vertex_biffurcation(self, vertex_idx, idx, adjacent_vertex_idx):
        source_vertex_idx = adjacent_vertex_idx
        sink_vertex_ids = self._edges[vertex_idx][:idx] + self._edges[vertex_idx][idx+1:]

        biffurcation_vertex_idx = vertex_idx

        self._biffurcations.append(
            TreeBiffurcation(
                source_vertex_idx=source_vertex_idx,
                biffurcation_vertex_idx=biffurcation_vertex_idx,
                sink_vertex_ids=sink_vertex_ids
            )
        )

    def __init_vertex_distance(self, vertex_idx, adjacent_vertex_idx):
        vertex = self.get_vertex(vertex_idx)
        adjacent_vertex = self.get_vertex(adjacent_vertex_idx)

        intervertex_distance = np.linalg.norm(vertex.point - adjacent_vertex.point)
        adjacent_vertex._distance = vertex.distance + intervertex_distance

    def __init_tree_attributes(self):
        pool = [ self._root_vertex_idx ]
        visited_vertex_ids = { self._root_vertex_idx }

        while pool:
            vertex_idx = pool[0]
            pool = pool[1:]

            for idx, adjacent_vertex_idx in enumerate(self._edges[vertex_idx]):
                if adjacent_vertex_idx not in visited_vertex_ids:
                    pool.append(adjacent_vertex_idx)

                    self.__init_vertex_distance(
                        vertex_idx,
                        adjacent_vertex_idx
                    )
                else:
                    if len(self._edges[vertex_idx]) > 2:
                        self.__init_vertex_biffurcation(
                            vertex_idx,
                            idx,
                            adjacent_vertex_idx
                        )

            visited_vertex_ids.add(vertex_idx)

    @property
    def nbiffurcations(self):
        return len(self._biffurcations)

    def get_biffurcation(self, idx):
        return self._biffurcations[idx]

    def __branch_iterator(self, previous_vertex_idx, next_vertex_idx):
        assert next_vertex_idx in self._edges[previous_vertex_idx]

        yield previous_vertex_idx

        while len(self._edges[next_vertex_idx]) == 2:
            yield next_vertex_idx

            if self._edges[next_vertex_idx][1] == previous_vertex_idx:
                vertex_idx = self._edges[next_vertex_idx][0]
            else:
                vertex_idx = self._edges[next_vertex_idx][1]

            previous_vertex_idx = next_vertex_idx
            next_vertex_idx = vertex_idx

        yield next_vertex_idx

    def _get_itermediate_vertex(self, first_vertex_idx, second_vertex_idx, ratio):
        assert first_vertex_idx in self._edges[second_vertex_idx]

        first_vertex = self.get_vertex(first_vertex_idx)
        second_vertex = self.get_vertex(second_vertex_idx)

        itermediate_point = first_vertex.point + ratio * (second_vertex.point - first_vertex.point)
        itermediate_distance = first_vertex.distance + ratio * (second_vertex.distance - first_vertex.distance)

        itermediate_vertex = TreeVertex(itermediate_point)
        itermediate_vertex._distance = itermediate_distance

        return itermediate_vertex

    def __step_branch_iterator(self, previous_vertex_idx, next_vertex_idx, step=0):
        assert step >= 0

        next_vertex_idx = self._edges[previous_vertex_idx][0]

        distance = self.get_vertex(previous_vertex_idx).distance

        for previous_vertex_idx, next_vertex_idx in pairwise(self.__branch_iterator(previous_vertex_idx, next_vertex_idx)):
            previous_vertex = self.get_vertex(previous_vertex_idx)
            next_vertex = self.get_vertex(next_vertex_idx)

            while round(distance, 3) < round(next_vertex.distance, 3):
                ratio = (distance - previous_vertex.distance) / (next_vertex.distance - previous_vertex.distance)
                yield self._get_itermediate_vertex(previous_vertex_idx, next_vertex_idx, ratio)

                distance += step if step > 0 else next_vertex.distance - previous_vertex.distance

        if np.isclose(distance, next_vertex.distance, atol=1e-03):
            yield self.get_vertex(next_vertex_idx)

    def aortic_iterator(self, *, step=0):
        previous_vertex_idx = self._root_vertex_idx

        if len(self._edges[previous_vertex_idx]) == 0:
            yield self._vertices[previous_vertex_idx]
        elif len(self._edges[previous_vertex_idx]) == 1:
            next_vertex_idx = self._edges[previous_vertex_idx][0]

            yield from self.__step_branch_iterator(previous_vertex_idx, next_vertex_idx, step)
        else:
            pass
