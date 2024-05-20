import pytest

import numpy as np

from aaa.geometry.getters.gettersVoxel import ( get_confidence, get_volume, get_volume_density,
                                                get_surface_square, get_intersection_surface_square )

@pytest.mark.getters
@pytest.mark.geometry
def test_get_confidence_CASE_default():
    probs = np.ones((3, 3, 3), bool)

    masks = np.zeros((3, 3, 3), bool)
    masks[:2, :2, :1] = 1

    output = get_confidence(probs, masks)['confidence']
    expected = 1

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_volume_CASE_default():
    masks = np.zeros((3, 3, 3), bool)
    masks[:2, :2, :1] = 1

    spacing = 3.

    output = get_volume(masks, spacing)['volume']
    expected = 4 * spacing**3

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_volume_density_CASE_default():
    imgs = 5 * np.ones((3, 3, 3))
    masks = np.zeros((3, 3, 3), bool)
    masks[:2, :2, :1] = 1

    spacing = 3.

    output = get_volume_density(imgs, masks, spacing)['mean volume density']
    expected = 5

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_surface_square_CASE_default():
    masks = np.ones((3, 3, 3), bool)

    spacing = 2.

    output = get_surface_square(masks, spacing)['surface square']
    expected = 216

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_surface_square_CASE_greater():
    masks = np.zeros((4, 4, 4), bool)
    masks[:3,:3,:3] = 1

    spacing = 1.

    output = get_surface_square(masks, spacing)['surface square']
    expected = 54

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_surface_square_CASE_smaller():
    masks = np.ones((2, 2, 3), bool)
    masks[:,:, 2] = 0

    spacing = 1.

    output = get_surface_square(masks, spacing)['surface square']
    expected = 24

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_surface_square_CASE_ignore_on_borders():
    masks = np.zeros((4, 4, 4), bool)
    masks[:3,:3,:3] = 1

    spacing = 1.

    output = get_surface_square(masks, spacing, ignore_on_borders=True)['surface square']
    expected = 27

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_intersection_surface_square_CASE_default():
    masks = np.ones((3, 3, 3), np.uint8)
    masks[:2,:2,:2] = 2

    spacing = 2.

    first_slices = (masks == 1)
    second_slices = (masks == 2)

    output = get_intersection_surface_square(first_slices, second_slices, spacing)['intersection surface square']
    expected = 48

    assert pytest.approx(output) == expected

@pytest.mark.getters
@pytest.mark.geometry
def test_get_intersection_surface_square_CASE_invariance_argument_order():
    masks = np.ones((3, 3, 3), np.uint8)
    masks[:2,:2,:2] = 2

    spacing = 2.

    first_slices = (masks == 1)
    second_slices = (masks == 2)

    output = get_intersection_surface_square(first_slices, second_slices, spacing)['intersection surface square']
    expected = get_intersection_surface_square(second_slices, first_slices, spacing)['intersection surface square']

    assert pytest.approx(output) == expected
