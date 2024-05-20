import numpy as np

from functools import partial
from multiprocessing import Pool
from collections import defaultdict

def __apply_image_augmentations_without_parallelization(imgs, masks, aug):
    auged_imgs = list()
    auged_masks = list()

    for img, mask in zip(imgs, masks):
        auged = aug(image=np.asarray(img), mask=np.asarray(mask))

        aug_img = auged['image']
        aug_mask = auged['mask']

        auged_imgs.append(aug_img)
        auged_masks.append(aug_mask)

    auged_imgs = np.array(auged_imgs)
    auged_imgs = np.moveaxis(auged_imgs, -1, 1)

    auged_masks = np.array(auged_masks, dtype=np.uint8)

    return auged_imgs, auged_masks

def __apply_image_augmentations_loop(args, aug):
    img, mask = args
    auged = aug(image=np.asarray(img), mask=np.asarray(mask))

    return auged['image'], auged['mask']

def __apply_image_augmentations_with_parallelization(imgs, masks, aug, *, njobs=1):
    args = zip(imgs, masks)
    loop = partial(__apply_image_augmentations_loop, aug=aug)

    pool = Pool(njobs)
    auged_imgs, auged_masks = zip(*pool.map(loop, args))

    auged_imgs = np.array(auged_imgs)
    auged_imgs = np.moveaxis(auged_imgs, -1, 1)

    auged_masks = np.array(auged_masks, dtype=np.uint8)

    return auged_imgs, auged_masks

def apply_image_augmentations(imgs, masks, aug, *, njobs=1):
    if njobs == 1:
        return __apply_image_augmentations_without_parallelization(imgs, masks, aug)
    else:
        return __apply_image_augmentations_with_parallelization(imgs, masks, aug, njobs=njobs)

def apply_voxel_augmentation(voxel, mask, aug, *, axis=None):
    additional_targets = dict()
    slices = dict()

    size = voxel.shape[axis]

    slicer = tuple(slice(0, None) for _ in np.arange(axis))
    zero_slicer = (*slicer, 0)

    for idx in np.arange(size):
        additional_targets[f'voxel_slice_{idx}'] = 'image'
        additional_targets[f'mask_slice_{idx}'] = 'mask'

        cslicer = (*slicer, idx)

        slices[f'voxel_slice_{idx}'] = voxel[cslicer]
        slices[f'mask_slice_{idx}'] = mask[cslicer]

    aug.add_targets(additional_targets)
    auged = aug( **slices,
                    image=voxel[zero_slicer],
                    mask=mask[zero_slicer] )

    auged_voxel = list()
    auged_mask = list()

    for idx in np.arange(size):
        additional_targets[f'voxel_slice_{idx}'] = None
        additional_targets[f'mask_slice_{idx}'] = None

        auged_voxel.append(auged[f'voxel_slice_{idx}'])
        auged_mask.append(auged[f'mask_slice_{idx}'])

    aug.add_targets(additional_targets)

    auged_voxel = np.stack(auged_voxel, axis=axis)
    auged_mask = np.stack(auged_mask, axis=axis)

    return auged_voxel, auged_mask

def apply_voxel_augmentations(voxels, masks, aug, *, axis=None):
    """ NOTE last dim treated as channel axis
    """
    auged_voxels = list()
    auged_masks = list()

    if axis is None:
        axis = 0
    else:
        assert axis >= 0 and axis < len(voxels.shape) - 1

    for voxel, mask in zip(voxels, masks):
        auged_voxel, auged_mask = apply_voxel_augmentation(voxel, mask, aug, axis=axis)

        auged_voxels.append(auged_voxel)
        auged_masks.append(auged_mask)

    auged_voxels = np.array(auged_voxels)
    auged_voxels = np.moveaxis(auged_voxels, -1, 1)

    auged_masks = np.array(auged_masks, dtype=np.uint8)

    return auged_voxels, auged_masks

class AlbumentationsMaskKeywordIgnorer(object):
    def __init__(self, aug):
        self.__aug = aug

    def add_targets(self, *args, **kwargs):
        return self.__aug.add_targets(*args, **kwargs)

    def __call__(self, **kwargs):
        filtred_kwargs = dict(kwargs)

        for key in kwargs.keys():
            if key.startswith('mask'):
                del filtred_kwargs[key]

        auged = defaultdict(lambda *args, **kwargs: 0)
        auged.update(self.__aug(**filtred_kwargs))

        return auged

class NoneSubscriptable(object):
    def __getitem__(self, *args, **kwargs):
        return None

def InfiniteNoneSubscriptableGenerator():
    while True:
        yield NoneSubscriptable()

def apply_only_image_augmentations(imgs, aug, *, njobs=1):
    masks = InfiniteNoneSubscriptableGenerator()
    aug = AlbumentationsMaskKeywordIgnorer(aug)

    auged_imgs, _ = apply_image_augmentations(imgs, masks, aug, njobs=njobs)

    return auged_imgs

def apply_only_voxel_augmentations(voxels, aug, *, axis=None):
    masks = InfiniteNoneSubscriptableGenerator()
    aug = AlbumentationsMaskKeywordIgnorer(aug)

    auged_voxels, _ = apply_voxel_augmentations(voxels, masks, aug, axis=axis)

    return auged_voxels
