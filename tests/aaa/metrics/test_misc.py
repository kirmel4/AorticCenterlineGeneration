
import pytest

import numpy as np

from aaa.metrics.misc import confusions

@pytest.mark.metrics
def test_confusions_CASE_one_label_type_AND_all_correct():
    true = np.array([[1, 0], [1, 0]])

    output = confusions(true, true)
    expected = { 0: { 'TP': 2,
                      'FP': 0,
                      'FN': 0 },
                 1: { 'TP': 2,
                      'FP': 0,
                      'FN': 0 },
    }

    assert output == expected

@pytest.mark.metrics
def test_confusions_CASE_one_label_type_AND_one_fp():
    true = np.array([[1, 0], [1, 0]])
    pred = np.array([[1, 1], [1, 0]])

    output = confusions(true, pred)
    expected = { 0: { 'TP': 1,
                      'FP': 0,
                      'FN': 1 },
                 1: { 'TP': 2,
                      'FP': 1,
                      'FN': 0 }
    }

    assert output == expected

@pytest.mark.metrics
def test_confusions_CASE_one_label_type_AND_one_fn():
    true = np.array([[1, 1], [1, 0]])
    pred = np.array([[1, 0], [1, 0]])

    output = confusions(true, pred)
    expected = { 0: { 'TP': 1,
                      'FP': 1,
                      'FN': 0 },
                 1: { 'TP': 2,
                      'FP': 0,
                      'FN': 1 }
    }

    assert output == expected

@pytest.mark.metrics
def test_confusions_CASE_two_label_type():
    true = np.array([[1, 1, 2], [1, 0, 0], [0, 2, 0]])
    pred = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    output = confusions(true, pred)
    expected = { 0: { 'TP': 1,
                      'FP': 2,
                      'FN': 3 },
                 1: { 'TP': 2,
                      'FP': 1,
                      'FN': 1 },
                 2: { 'TP': 1,
                      'FP': 2,
                      'FN': 1 }
    }

    assert output == expected
