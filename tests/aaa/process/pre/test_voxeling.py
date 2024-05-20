import pytest

import numpy as np

from aaa.process.pre.voxeling import voxel_sequential_selector

@pytest.mark.pre
@pytest.mark.process
def test_voxel_sequential_selector_CASE_aliquot_steps():
    case_keys = ['1']
    voxel_shape = (2, 2, 2)
    shapes = [(4, 2, 2)]

    steps = (2, 2, 2)

    output = [* voxel_sequential_selector(voxel_shape, case_keys, shapes, steps) ]

    expected = [
                ('1', ( slice(0, 2, None),
                        slice(0, 2, None),
                        slice(0, 2, None))),
                ('1', ( slice(2, 4, None),
                        slice(0, 2, None),
                        slice(0, 2, None))),
            ]

    assert output == expected

@pytest.mark.pre
@pytest.mark.process
def test_voxel_sequential_selector_CASE_unaliquot_steps():
    case_keys = ['1']
    voxel_shape = (3, 2, 2)
    shapes = [(6, 2, 2)]

    steps = (2, 2, 2)

    output = [* voxel_sequential_selector(voxel_shape, case_keys, shapes, steps) ]

    expected = [
                ('1', ( slice(0, 3, None),
                        slice(0, 2, None),
                        slice(0, 2, None))),
                ('1', ( slice(2, 5, None),
                        slice(0, 2, None),
                        slice(0, 2, None))),
                ('1', ( slice(3, 6, None),
                        slice(0, 2, None),
                        slice(0, 2, None))),
            ]

    assert output == expected
