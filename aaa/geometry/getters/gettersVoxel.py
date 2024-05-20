import numpy as np

from scipy import ndimage
from functools import partial

from aaa.utils.decorators import check_arguments

def __binary_asserter(args):
    assert args['slices'].dtype == bool

@partial(check_arguments, checkers=[__binary_asserter])
def get_confidence(probslices, slices):
    confidence = np.mean(probslices[slices])

    return { 'confidence': float(confidence) }

@partial(check_arguments, checkers=[__binary_asserter])
def get_volume(slices, spacing):
    volume = np.sum(slices) * spacing**3

    return { 'volume': float(volume),
             'volume unit': 'mm^3' }

@partial(check_arguments, checkers=[__binary_asserter])
def get_volume_density(imgslices, slices, spacing):
    masses = imgslices[slices]
    volumes = slices[slices]

    if np.sum(volumes) > 0:
        mean_volume_density = np.sum(masses) / np.sum(volumes)
        std_volume_density = np.std(masses / volumes)
        volume_densities = (masses / volumes).tolist()
    else:
        mean_volume_density = 0.
        std_volume_density = 0.
        volume_densities = list()

    return { 'mean volume density': float(mean_volume_density),
             'std volume density': float(std_volume_density),
             'volume densities': volume_densities,
             'volume density unit': 'HU per mm^3' }

def __get_connection_matrix():
    kernel = np.zeros((3, 3, 3), dtype=np.uint8)

    for i, j, k in [ (0, 1, 1), (1, 0, 1),
                     (1, 1, 0), (1, 1, 2),
                     (1, 2, 1), (2, 1, 1), ]:
        kernel[i, j, k] = 1

    return kernel

@partial(check_arguments, checkers=[__binary_asserter])
def get_surface_square(slices, spacing, ignore_on_borders=False):
    kernel = __get_connection_matrix()

    if ignore_on_borders:
        mode = 'nearest'
    else:
        mode = 'constant'

    cslices = ndimage.convolve( slices.astype(np.uint8),
                                kernel,
                                mode=mode,
                                cval=0 )

    voxels_on_edge = cslices[slices & (cslices > 0)]
    surface_square = np.sum(np.sum(kernel) - voxels_on_edge) * spacing**2

    return { 'surface square': float(surface_square),
             'square unit': 'mm^2' }

@partial(check_arguments, checkers=[__binary_asserter])
def get_intersection_surface_square(slices, islices, spacing, ignore_on_borders=False):
    kernel = __get_connection_matrix()

    if ignore_on_borders:
        mode = 'nearest'
    else:
        mode = 'constant'

    cslices = ndimage.convolve( slices.astype(np.uint8),
                                kernel,
                                mode=mode,
                                cval=0 )

    voxels_on_edge = cslices[islices & (cslices > 0)]
    intersection_surface_square = np.sum(voxels_on_edge) * spacing**2

    return { 'intersection surface square': float(intersection_surface_square),
             'square unit': 'mm^2' }
