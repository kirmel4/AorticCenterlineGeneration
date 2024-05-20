import numpy as np

from scipy.spatial import KDTree
from cc3d import connected_components
from scipy.ndimage import binary_fill_holes

from aaa.geometry.enum import Classes

def filter_largest_connected_component(masks, connectivity):
    filter_masks = np.array(masks)
    booled_masks = (masks > 0).astype(np.uint8)

    components, ncomponents = connected_components(booled_masks, connectivity=connectivity, return_N=True)

    if ncomponents > 0:
        sizes = [ np.sum(components == idx + 1) for idx in np.arange(ncomponents) ]
        filter_masks[components != (np.argmax(sizes) + 1)] = 0

    return filter_masks

def filter_holes(masks, ncands=1):
    filtred_masks = np.array(masks)

    booled_masks = (masks > 0).astype(np.bool_)
    booled_masks_without_holes = binary_fill_holes(booled_masks)
    holes = np.logical_xor(booled_masks, booled_masks_without_holes)

    if np.sum(holes):
        xcoord, ycoord, zcoord = np.nonzero(booled_masks)
        xhcoord, yhcoord, zhcoord = np.nonzero(holes)

        xyz_without_holes = np.array((xcoord, ycoord, zcoord)).T
        xyz_holes = np.array((xhcoord, yhcoord, zhcoord)).T

        tree = KDTree(xyz_without_holes)

        candidates = masks[booled_masks][tree.query(xyz_holes, k=ncands)[1]]

        if ncands == 1:
            candidates = candidates[:, None]

        votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(Classes)), 1, candidates)
        final_candidates = np.argmax(votes, axis=1)

        filtred_masks[holes] = final_candidates
    else:
        filtered_masks = masks

    return filtred_masks
