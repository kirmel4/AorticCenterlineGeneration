import numpy as np

from aaa.geometry.enum import Classes
from aaa.geometry.getters.gettersMesh import get_surface_square
from aaa.geometry.getters.getters import get_surface_ratio_rate
from aaa.geometry.descriptors.descriptor3d import Descriptor3D

class DescriptorEVAR(Descriptor3D):
    def __init__(self, imgs, probs, *, spacing=(1., 1., 1.), nspacing=(1., 1., 1.)):
        Descriptor3D.__init__(self, imgs, probs, spacing=spacing, nspacing=nspacing)

    def get_surface_ratio_rate(self):
        inner_surface_square = get_surface_square(self.inner_surface, self._spacing)['surface square']
        outer_surface_square = get_surface_square(self.outer_surface, self._spacing)['surface square']

        return { 'surface ratio rate': get_surface_ratio_rate(
                                        inner_surface_square,
                                        outer_surface_square
                                       )['surface ratio rate'] }
