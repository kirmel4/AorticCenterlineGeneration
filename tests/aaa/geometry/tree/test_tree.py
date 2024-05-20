import pytest

import numpy as np

from aaa.geometry.tree.tree import Tree
from aaa.geometry.tree.tree_errors import TreeConstructionError

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_zero_point():
    z = []
    x = []
    y = []

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    with pytest.raises(TreeConstructionError, match='point list is empty') as e:
        tree = Tree(points)

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_one_point_AND_incorrect_large_root_index():
    z = [0]
    x = [0]
    y = [0]

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    with pytest.raises(TreeConstructionError, match='root index is out point range') as e:
        tree = Tree(points, root_idx=1)

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_one_point_AND_incorrect_negative_root_index():
    z = [0]
    x = [0]
    y = [0]

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    with pytest.raises(TreeConstructionError, match='root index is out point range') as e:
        tree = Tree(points, root_idx=-1)

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_one_point():
    z = [0]
    x = [0]
    y = [0]

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator()]) == 1

    vertex = tree.get_vertex(0)

    assert pytest.approx(vertex.point) == [0, 0, 0,]
    assert pytest.approx(vertex.distance) == 0.

    assert tree.nbiffurcations == 0

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_two_point():
    z = [0, 1]
    x = [0, 0]
    y = [0, 0]

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator()]) == 2

    for vdx, vertex in enumerate(tree.aortic_iterator()):
        print(vertex.distance)

        assert pytest.approx(vertex.point) == points[vdx]
        assert pytest.approx(vertex.distance) == vdx * 1

    assert tree.nbiffurcations == 0

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_four_points_AND_one_biffurcation():
    z = [0, 1, 1, 1]
    x = [0, 0, 1, -1]
    y = [0, 0, 0, 0]

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator()]) == 2

    for vdx, vertex in enumerate(tree.aortic_iterator()):
        assert pytest.approx(vertex.point) == points[vdx]

    assert tree.nbiffurcations == 1

    source_idx = tree.get_biffurcation(0)._source_vertex_idx
    source_vertex = tree.get_vertex(source_idx)

    assert pytest.approx(source_vertex.point) == [0, 0, 0]
    assert pytest.approx(source_vertex.distance) == 0.

    biffurcation_idx = tree.get_biffurcation(0)._biffurcation_vertex_idx
    biffurcation_vertex = tree.get_vertex(biffurcation_idx)

    assert pytest.approx(biffurcation_vertex.point) == [1, 0, 0]
    assert pytest.approx(biffurcation_vertex.distance) == 1.

    sink_ids = tree.get_biffurcation(0)._sink_vertex_ids

    for vertex_idx in sink_ids:
        vertex = tree.get_vertex(vertex_idx)

        assert pytest.approx(np.abs(vertex.point)) == [1, 1, 0]
        assert pytest.approx(vertex.distance) == 2.

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_segment_shape():
    SIZE = 100

    t = np.linspace(0, 1, SIZE)

    z = t
    x = np.zeros(SIZE)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    pids = np.arange(1 , SIZE)
    np.random.shuffle(pids)
    pids = np.insert(pids, 0, 0)

    tree = Tree(points[pids])

    assert len([*tree.aortic_iterator()]) == SIZE

    for vdx, vertex in enumerate(tree.aortic_iterator()):
        assert pytest.approx(vertex.point) == points[vdx]
        assert pytest.approx(vertex.distance) == vdx / (SIZE-1)

    assert tree.nbiffurcations == 0

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_semicircle_shape():
    SIZE = 100

    t = np.linspace(0, 1, SIZE)

    z = np.cos(np.pi * t)
    x = np.sin(np.pi * t)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    distances = np.cumsum(
        np.linalg.norm(
            np.diff(points, axis=0),
            axis=1
        )
    )

    distances = np.insert(distances, 0, 0)

    pids = np.arange(1 , SIZE)
    np.random.shuffle(pids)
    pids = np.insert(pids, 0, 0)

    tree = Tree(points[pids])

    assert len([*tree.aortic_iterator()]) == SIZE

    for vdx, vertex in enumerate(tree.aortic_iterator()):
        assert pytest.approx(vertex.point) == points[vdx]
        assert pytest.approx(vertex.distance) == distances[vdx]

    assert tree.nbiffurcations == 0

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_creation_AND_star_shape():
    SIZE = 100

    t = np.linspace(0, 1, SIZE)

    zs1 = t
    xs1 = np.zeros(SIZE)
    ys1 = np.zeros(SIZE)

    zs2 = 1 + t[1:]
    xs2 = t[1:]
    ys2 = np.zeros(SIZE-1)

    zs3 = 1 + t[1:]
    xs3 = -t[1:]
    ys3 = np.zeros(SIZE-1)

    z = [*zs1, *zs2, *zs3]
    x = [*xs1, *xs2, *xs3]
    y = [*ys1, *ys2, *ys3]

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    pids = np.arange(1 , 3 * SIZE - 2)
    np.random.shuffle(pids)
    pids = np.insert(pids, 0, 0)

    tree = Tree(points[pids])

    for vdx, vertex in enumerate(tree.aortic_iterator()):
        assert pytest.approx(vertex.point) == points[vdx]
        assert pytest.approx(vertex.distance) == vdx / (SIZE-1)

    assert tree.nbiffurcations == 1

    source_idx = tree.get_biffurcation(0)._source_vertex_idx
    source_vertex = tree.get_vertex(source_idx)

    assert pytest.approx(source_vertex.point) == [1-1/(SIZE-1), 0, 0]

    biffurcation_idx = tree.get_biffurcation(0)._biffurcation_vertex_idx
    biffurcation_vertex = tree.get_vertex(biffurcation_idx)

    assert pytest.approx(biffurcation_vertex.point) == [1, 0, 0]

    sink_ids = tree.get_biffurcation(0)._sink_vertex_ids

    for vertex_idx in sink_ids:
        vertex = tree.get_vertex(vertex_idx)

        assert pytest.approx(np.abs(vertex.point)) == [1+1/(SIZE-1), 1/(SIZE-1), 0]
        assert pytest.approx(vertex.distance) == 1 + 2**0.5/(SIZE-1)

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_aortic_iteration_AND_zero_step():
    SIZE = 11
    STEP = 0

    t = np.linspace(0, 1, SIZE)

    z = t
    x = np.zeros(SIZE)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator(step=STEP)]) == SIZE

    for vdx, vertex in enumerate(tree.aortic_iterator(step=STEP)):
        assert pytest.approx(vertex.point) == [vdx / (SIZE - 1), 0, 0]

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_aortic_iteration_AND_zero_point_one_step():
    SIZE = 11
    STEP = 0.1

    t = np.linspace(0, 1, SIZE)

    z = t
    x = np.zeros(SIZE)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator()]) == 11

    for vdx, vertex in enumerate(tree.aortic_iterator(step=STEP)):
        assert pytest.approx(vertex.point) == [vdx / (SIZE - 1), 0, 0]

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_aortic_iteration_AND_zero_point_two_step():
    SIZE = 11
    STEP = 0.2

    t = np.linspace(0, 1, SIZE)

    z = t
    x = np.zeros(SIZE)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator(step=STEP)]) == 6

    for vdx, vertex in enumerate(tree.aortic_iterator(step=STEP)):
        assert pytest.approx(vertex.point) == [ 2 * vdx / (SIZE - 1), 0, 0]


@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_aortic_iteration_AND_zero_point_three_step():
    SIZE = 11
    STEP = 0.3

    t = np.linspace(0, 1, SIZE)

    z = t
    x = np.zeros(SIZE)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator(step=STEP)]) == 4

    for vdx, vertex in enumerate(tree.aortic_iterator(step=STEP)):
        assert pytest.approx(vertex.point) == [ 3 * vdx / (SIZE - 1), 0, 0]

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_aortic_iteration_AND_zero_point_two_five_step():
    SIZE = 11
    STEP = 0.25

    t = np.linspace(0, 1, SIZE)

    z = t
    x = np.zeros(SIZE)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator(step=STEP)]) == 5

    for vdx, vertex in enumerate(tree.aortic_iterator(step=STEP)):
        assert pytest.approx(vertex.point) == [ 2.5 * vdx / (SIZE - 1), 0, 0]

@pytest.mark.tree
@pytest.mark.geometry
def test_Tree_CASE_aortic_iteration_AND_zero_point_zero_five_step():
    SIZE = 11
    STEP = 0.05

    t = np.linspace(0, 1, SIZE)

    z = t
    x = np.zeros(SIZE)
    y = np.zeros(SIZE)

    points = np.array([z, x, y])
    points = np.moveaxis(points, 0, -1)

    tree = Tree(points)

    assert len([*tree.aortic_iterator(step=STEP)]) == 21

    for vdx, vertex in enumerate(tree.aortic_iterator(step=STEP)):
        assert pytest.approx(vertex.point) == [ 0.5 * vdx / (SIZE - 1), 0, 0]
