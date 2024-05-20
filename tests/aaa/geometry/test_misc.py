import pytest

import cv2
import numpy as np

from aaa.geometry.misc import reshape, respace, __respace_iterate_zero_axis, __respace_iterate_second_axis

INTERPOLATIONS = [ cv2.INTER_LINEAR,
                   cv2.INTER_NEAREST,
                   cv2.INTER_CUBIC,
                   cv2.INTER_LANCZOS4 ]

@pytest.fixture()
def respace_factors():
     xspacing = np.random.random(1)[0] * 0.5 + 0.5
     yspacing = np.random.random(1)[0] * 0.5 + 0.5
     thickness = np.random.random(1)[0] * 0.5 + 0.5

     xspacing = np.round(xspacing, 3)
     yspacing = np.round(yspacing, 3)
     thickness = np.round(thickness, 3)

     return (yspacing, xspacing, thickness)

@pytest.mark.geometry
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_respace_CASE_invariance_resample_order(respace_factors, interpolation):
     nslices = np.random.randint(32, 256)
     data = np.random.randint(0, 255, size=(nslices, 512, 512)).astype('float64')

     yspacing, xspacing, thickness = respace_factors

     data_one_order = __respace_iterate_zero_axis(data, xspacing, yspacing, interpolation)
     data_one_order = __respace_iterate_second_axis(data_one_order, 1., thickness, interpolation)

     data_two_order = __respace_iterate_second_axis(data, yspacing, thickness, interpolation)
     data_two_order = __respace_iterate_zero_axis(data_two_order, xspacing, 1., interpolation)

     assert np.allclose(data_one_order, data_two_order, atol=1e-5)

@pytest.mark.geometry
def test_respace_CASE_default():
    data = np.ones((40, 30, 20))

    spacing = (1., 1., 1.)
    nspacing = (4., 3., 2.)

    ndata = respace(data, spacing, nspacing)

    expected_shape = (10, 10, 10)

    assert ndata.shape == expected_shape

@pytest.mark.geometry
def test_reshape_CASE_default():
    data = np.ones((40, 30, 20))

    shape = (40, 30, 20)
    nshape = (10, 10, 10)

    ndata = reshape(data, shape, nshape)

    expected_shape = (10, 10, 10)

    assert ndata.shape == expected_shape
