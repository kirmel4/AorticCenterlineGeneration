import pytest

from aaa.utils.itertools import unchain, pairwise

@pytest.mark.utils
def test_unchain_CASE_default():
    input_ = range(5)

    output = [* unchain(input_, 3)]

    expected = [ [0, 1, 2], [3, 4] ]

    assert output ==  expected

@pytest.mark.utils
def test_pairwise_CASE_default():
    input_ = range(3)

    output = [* pairwise(input_) ]

    expected = [ (0, 1), (1, 2) ]

    assert output ==  expected
