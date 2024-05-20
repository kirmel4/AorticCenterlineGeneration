import pytest

import numpy as np

from aaa.process.post.filtration import filter_largest_connected_component, filter_holes

@pytest.mark.post
@pytest.mark.process
def test_filter_largest_connected_component_CASE_default():
    masks = np.ones((4, 4, 4), dtype=np.int32)
    masks[:2, :2, :2] = 0
    masks[0, 0, 0] = 1
    masks[2, 2, 2] = 1

    output_component = filter_largest_connected_component(masks, connectivity=6)
    expected_component = np.ones((4, 4, 4), dtype=np.int32)
    expected_component[:2, :2, :2] = 0
    expected_component[2, 2, 2] = 1

    assert np.all(output_component == expected_component)

@pytest.mark.post
@pytest.mark.process
def test_filter_largest_connected_component_CASE_empty_mask():
    masks = np.zeros((4, 4, 4), dtype=np.int32)

    output_component = filter_largest_connected_component(masks, connectivity=6)
    expected_component = np.zeros((4, 4, 4), dtype=np.int32)

    assert np.all(output_component == expected_component)

@pytest.mark.post
@pytest.mark.process
def test_filter_holes_CASE_one_label_AND_one_candidate():
    masks = np.ones((3, 3, 3), dtype=np.int32)
    masks[1, 1, 1] = 0

    output = filter_holes(masks)
    expected = np.ones((3, 3, 3), dtype=np.int32)

    assert np.all(output == expected)

@pytest.mark.post
@pytest.mark.process
def test_filter_holes_CASE_two_labels_AND_one_candidate():
    masks = np.ones((3, 3, 3), dtype=np.int32)
    masks[1, 1, 1] = 0
    masks[2, 2, 2] = 1

    output = filter_holes(masks)
    expected = np.ones((3, 3, 3), dtype=np.int32)
    expected[2, 2, 2] = 1

    assert np.all(output == expected)

@pytest.mark.post
@pytest.mark.process
def test_filter_holes_CASE_two_labels_AND_six_candidates_AND_labels_one_two():
    mask_layer = np.ones((3, 3), dtype=np.int32)
    mask_layer[1, 1] = 2

    mask_layer_middle = np.ones((3, 3), dtype=np.int32)

    masks = np.array([mask_layer, mask_layer_middle, mask_layer])
    masks[1, 1, 1] = 0

    output = filter_holes(masks, ncands=6)
    expected = np.array(masks)
    expected[1, 1, 1] = 1

    assert np.all(output == expected)

@pytest.mark.post
@pytest.mark.process
def test_filter_holes_CASE_two_labels_AND_six_candidates_AND_labels_one_three():
    mask_layer = np.ones((3, 3), dtype=np.int32)
    mask_layer[1, 1] = 3

    mask_layer_middle = 3 * np.ones((3, 3), dtype=np.int32)

    masks = np.array([mask_layer, mask_layer_middle, mask_layer])
    masks[1, 1, 1] = 0

    output = filter_holes(masks, ncands=6)
    expected = np.array(masks)
    expected[1, 1, 1] = 3

    assert np.all(output == expected)

@pytest.mark.post
@pytest.mark.process
def test_filter_holes_CASE_four_labels_AND_two_holes():
    mask_layer_first_hole = np.ones((3, 3), dtype=np.int32)
    mask_layer_first_hole[1, 1] = 3

    mask_layer_middle_first_hole = 3 * np.ones((3, 3), dtype=np.int32)
    mask_layer_middle_first_hole[1, 1] = 0

    mask_layer_pad = np.zeros((3, 3), dtype=np.int32)

    mask_layer_second_hole = np.ones((3, 3), dtype=np.int32)
    mask_layer_second_hole[1, 1] = 2

    mask_layer_middle_second_hole = 2 * np.ones((3, 3), dtype=np.int32)
    mask_layer_middle_second_hole[1, 1] = 0

    masks = np.array([ mask_layer_first_hole,
                       mask_layer_middle_first_hole,
                       mask_layer_first_hole,
                       mask_layer_pad,
                       mask_layer_second_hole,
                       mask_layer_middle_second_hole,
                       mask_layer_second_hole
                    ])

    output = filter_holes(masks, ncands=6)
    expected = np.array(masks)
    expected[1, 1, 1] = 3
    expected[5, 1, 1] = 2

    assert np.all(output == expected)
