import cv2
import numpy as np

from aaa.augmentations.domain_adaptation import FiltredBackProjectionNormalization
from aaa.process.pre import apply_only_image_augmentations

def drop_empty_slices(images, masks, *, range=None):
    if range is None:
        masks_not_null = np.sum(masks, axis=(1, 2))
        ids, = np.where(masks_not_null > 0)
        range = ids
    else:
        range = slice(*range)

    images = images[range, :, :]
    masks = masks[range, :, :]

    return images, masks

def drop_empty_slices_with_probs(images, masks, probs, *, range=None):
    if range is None:
        masks_not_null = np.sum(masks, axis=(1, 2))
        ids, = np.where(masks_not_null > 0)
        range = ids
    else:
        range = slice(*range)

    images = images[range, :, :]
    masks = masks[range, :, :]
    probs = probs[range, :, :]

    return images, masks, probs

def normalize_HU(images, *, MIN_HU, MAX_HU):
    images = (images - MIN_HU) / (MAX_HU - MIN_HU)
    return images.clip(min=0., max=1.)

def normalize_reconstruction_filter(images, njobs=1):
    aug = FiltredBackProjectionNormalization()

    images = apply_only_image_augmentations(images, aug=aug, njobs=njobs)
    return images

def split_images(images, channels):
    image_channels = ()

    for options in channels.values():
        image_channel = normalize_HU(images, MIN_HU=options['MIN_HU'], MAX_HU=options['MAX_HU'])
        image_channels = (*image_channels, image_channel)

    return np.concatenate(image_channels, axis=-1)

def normalize_outside_pixels(images):
    Air_HU = -1000

    nrows, ncols, _ = images.shape
    assert nrows == ncols

    radius = nrows // 2
    xpr, ypr = np.mgrid[:nrows, :ncols] - radius
    outside_pixel_mask = (xpr ** 2 + ypr ** 2) > radius ** 2
    images[outside_pixel_mask] = Air_HU

    return images
