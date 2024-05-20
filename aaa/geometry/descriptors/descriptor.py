import cv2
import math
import numpy as np

from aaa.geometry.enum import Classes
from aaa.geometry.misc import respace
from aaa.geometry.getters.gettersVoxel import ( get_confidence, get_volume, get_volume_density,
                                                get_surface_square, get_intersection_surface_square )
from aaa.process.post import filter_largest_connected_component, filter_holes


class Descriptor(object):
    def __init__(self, imgs, probs, *, spacing=(1., 1., 1.), nspacing=(1., 1., 1.)):
        """
        Data in zxy(channel)

        :NOTE:
            xspacing equals yspacing

        :args:
            imgs (np.array[z, x, y]): array with a CT volume
            probs (np.array[z, x, y, ch]): array with segmentation probabilities
            spacing (list of (float, float, float)): original spacing of the arrays
            nspacing (list of (float, float, float)): target spacing to the arrays
        """

        assert math.isclose(spacing[1], spacing[2])

        self._imgs = respace(imgs, spacing, nspacing, cv2.INTER_LINEAR)

        probs = probs.astype(np.float64)
        self._probs = list()

        for c in Classes:
            self._probs.append(respace(probs[:,:,:,c.value], spacing, nspacing, cv2.INTER_LINEAR))

        self._probs = np.stack(self._probs, axis=-1)
        self._masks = self._probs.argmax(-1).astype(np.uint8)

        self._masks = filter_largest_connected_component(self._masks, connectivity=18)
        self._masks = filter_holes(self._masks, ncands=3)

        self._spacing = nspacing[0]

    def get_confidence(self, *, rkey='full', zrange=(None, -1)):
        confidence = { }

        for class_ in Classes:
            if class_.value == 0:
                continue

            key = ' '.join(class_.name.lower().split('_'))

            masks = self._masks[slice(*zrange)] == class_.value
            probs = self._probs[slice(*zrange), :, :, class_.value]

            info = get_confidence(probs, masks)
            confidence[f'{rkey} {key} confidence'] = info['confidence']

        return confidence

    def get_volumes(self, *, rkey='full', zrange=(None, -1)):
        volumes = { 'volume unit': 'mm^3' }

        for class_ in Classes:
            if class_.value == 0:
                continue

            key = ' '.join(class_.name.lower().split('_'))

            masks = self._masks[slice(*zrange)] == class_.value

            info = get_volume(masks, self._spacing)
            volumes[f'{rkey} {key} volume'] = info['volume']

        return volumes

    def get_densities(self, *, rkey='full', zrange=(None, -1)):
        densities = { 'volume density unit': 'HU per mm^3' }

        for class_ in Classes:
            if class_.value == 0:
                continue

            key = ' '.join(class_.name.lower().split('_'))

            imgs = self._imgs[slice(*zrange)]
            masks = self._masks[slice(*zrange)] == class_.value

            info = get_volume_density(imgs, masks, self._spacing)
            densities[f'{rkey} {key} mean volume density'] = info['mean volume density']
            densities[f'{rkey} {key} std volume density'] = info['std volume density']

        return densities

    def get_surfaces(self, *, rkey='full', zrange=(None, -1)):
        surfaces = { 'surface unit': 'mm^2' }

        lumen_masks = self._masks[slice(*zrange)] == Classes.LUMEN.value
        calcifications_masks = self._masks[slice(*zrange)] == Classes.CALCIFICATIONS.value

        info = get_surface_square(lumen_masks, self._spacing)
        surfaces[f'{rkey} lumen surface square'] = info['surface square']

        info = get_intersection_surface_square(lumen_masks, calcifications_masks, self._spacing)
        surfaces[f'{rkey} lumen and calcifications intersection surface square'] = info['intersection surface square']

        return surfaces
