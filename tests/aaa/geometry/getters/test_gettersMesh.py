import pytest

import numpy as np

from aaa.geometry.trimesh import get_marching_cubes_surface
from aaa.geometry.getters.gettersVoxel import ( get_surface_square as get_surface_square_on_voxel )
from aaa.geometry.getters.gettersMesh import ( get_surface_square as get_surface_square_on_mesh )

@pytest.mark.getters
@pytest.mark.geometry
def test_get_surface_square_CASE_default():
    SPACING = 0.8

    masks = np.zeros((60, 60, 60), bool)
    masks[20:40, 20:40, 20:40] = 1

    mesh = get_marching_cubes_surface(masks)

    output = get_surface_square_on_mesh(mesh, SPACING)['surface square']
    expected = get_surface_square_on_voxel(masks, SPACING)['surface square']

    error = 100 * np.abs(1 - output / expected)

    assert error < 5
