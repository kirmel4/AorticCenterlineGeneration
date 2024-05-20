import pytest

import numpy as np
import albumentations as A

from aaa.process.pre.augmentations import apply_image_augmentations, apply_voxel_augmentations

SIZE = 4

@pytest.fixture()
def aug():
    return A.Compose([ A.CenterCrop( height=SIZE,
                                     width=SIZE,
                                     p=1.0 )
                      ])

@pytest.mark.pre
@pytest.mark.process
def test_apply_image_augmentations_CASE_default(aug):
    center = slice(SIZE // 2, -SIZE // 2)

    imgs = np.ones((2, 2 * SIZE, 2 * SIZE, 1))
    imgs[0, center, center] = 2
    imgs[1, center, center] = 3

    masks = np.zeros((2, 2 * SIZE, 2 * SIZE))
    masks[:, center, center] = 1

    auged_imgs, auged_masks = apply_image_augmentations(imgs, masks, aug)

    expected_imgs = np.ones((2, 1 , SIZE, SIZE))
    expected_imgs[0] *= 2
    expected_imgs[1] *= 3

    expected_masks = np.ones((2 , SIZE, SIZE))

    assert np.all(auged_imgs == expected_imgs)
    assert np.all(auged_masks == expected_masks)

@pytest.mark.pre
@pytest.mark.process
def test_apply_voxel_augmentations_CASE_one_channel(aug):
    center = slice(SIZE // 2, -SIZE // 2)

    voxels = np.ones((2, SIZE, 2 * SIZE, 2 * SIZE, 1))
    voxels[0, :, center, center] = 2
    voxels[1, :, center, center] = 3

    masks = np.zeros((2, SIZE, 2 * SIZE, 2 * SIZE))
    masks[:, :, center, center] = 1

    auged_voxels, auged_masks = apply_voxel_augmentations(voxels, masks, aug)

    expected_voxels = np.ones((2, 1, SIZE , SIZE, SIZE))
    expected_voxels[0] *= 2
    expected_voxels[1] *= 3

    expected_masks = np.ones((2, SIZE, SIZE, SIZE))

    assert auged_voxels.shape == (2, 1, SIZE, SIZE, SIZE)
    assert auged_masks.shape == (2, SIZE, SIZE, SIZE)

    assert np.all(auged_voxels == expected_voxels)
    assert np.all(auged_masks == expected_masks)

@pytest.mark.pre
@pytest.mark.process
def test_apply_voxel_augmentations_CASE_two_channel(aug):
    center = slice(SIZE // 2, -SIZE // 2)

    voxels = np.ones((2, SIZE, 2 * SIZE, 2 * SIZE, 2))
    voxels[0, :, center, center, 0] = 2
    voxels[1, :, center, center, 0] = 3

    masks = np.zeros((2, SIZE, 2 * SIZE, 2 * SIZE))
    masks[:, :, center, center] = 1

    auged_voxels, auged_masks = apply_voxel_augmentations(voxels, masks, aug)

    expected_voxels = np.ones((2, 2, SIZE , SIZE, SIZE))
    expected_voxels[0, 0] *= 2
    expected_voxels[1, 0] *= 3

    expected_masks = np.ones((2, SIZE, SIZE, SIZE))

    assert auged_voxels.shape == (2, 2, SIZE, SIZE, SIZE)
    assert auged_masks.shape == (2, SIZE, SIZE, SIZE)

    assert np.all(auged_voxels == expected_voxels)
    assert np.all(auged_masks == expected_masks)

@pytest.mark.pre
@pytest.mark.process
def test_apply_voxel_augmentations_CASE_nonzero_axis(aug):
    center = slice(SIZE // 2, -SIZE // 2)

    voxels = np.ones((2, 2 * SIZE, SIZE, 2 * SIZE, 1))
    voxels[0, center, :, center] = 2
    voxels[1, center, :, center] = 3

    masks = np.zeros((2, 2 * SIZE, SIZE, 2 * SIZE))
    masks[:, center, :, center] = 1

    auged_voxels, auged_masks = apply_voxel_augmentations(voxels, masks, aug, axis=1)

    expected_voxels = np.ones((2, 1, SIZE , SIZE, SIZE))
    expected_voxels[0] *= 2
    expected_voxels[1] *= 3

    expected_masks = np.ones((2, SIZE, SIZE, SIZE))

    assert auged_voxels.shape == (2, 1, SIZE, SIZE, SIZE)
    assert auged_masks.shape == (2, SIZE, SIZE, SIZE)

    assert np.all(auged_voxels == expected_voxels)
    assert np.all(auged_masks == expected_masks)

@pytest.mark.pre
@pytest.mark.process
def test_apply_voxel_augmentations_CASE_wrong_axis(aug):
    voxels = np.ones((2, SIZE, 2 * SIZE, 2 * SIZE, 1))
    masks = np.zeros((2, SIZE, 2 * SIZE, 2 * SIZE))

    with pytest.raises(Exception):
        apply_voxel_augmentations(voxels, masks, aug, axis=3)
