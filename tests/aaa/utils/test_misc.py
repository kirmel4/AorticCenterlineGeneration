import pytest

import numpy as np

from aaa.utils.misc import batch_ids_generator

@pytest.mark.utils
def test_batch_ids_generator_CASE_default():
    output = batch_ids_generator(10, 3)
    expected = [ np.array([0, 1, 2]),
                 np.array([3, 4, 5]),
                 np.array([6, 7, 8]),
                 np.array([9]) ]

    assert len(output) == len(expected)

    for out, exp in zip(output, expected):
        assert np.all(out == exp)

@pytest.mark.utils
def test_batch_ids_generator_CASE_size_smaller_than_batch_size():
    output = batch_ids_generator(2, 3)
    expected = [ np.array([0, 1]) ]

    assert len(output) == len(expected)

    for out, exp in zip(output, expected):
        assert np.all(out == exp)

@pytest.mark.utils
def test_batch_ids_generator_CASE_shuffle():
    output = batch_ids_generator(10, 3, shuffle=True)

    len_expected = 4
    size_expected = 3

    assert len(output) == len_expected

    for out in output[:-1]:
        assert len(out) == size_expected
