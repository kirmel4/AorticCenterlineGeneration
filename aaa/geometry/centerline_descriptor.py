# import cv2
# import cc3d
# import math
# import numpy as np

# from scipy.ndimage import map_coordinates

# from aaa.geometry.misc import respace
# from aaa.utils.decorators import fields
# from aaa.geometry.constants import AIR_HU
# from aaa.geometry.enum import Classes
# from aaa.process.post import get_maximal_connected_component
# from aaa.geometry.centerline_based.misc import get_volume, get_volume_density

# class CenterlineDescriptor(object):
#     def __init__(self, imgs, probs, *, spacing=(1., 1., 1.), nspacing=(0.5, 0.5, 0.5)):
#         """
#         Data in zxy(channel)

#         None xspacing equals yspacing
#         """
#         assert math.isclose(spacing[1], spacing[2])

#         self._imgs = respace(imgs, spacing, nspacing, cv2.INTER_LINEAR)

#         probs = probs.astype(np.float64)
#         self._probs = list()

#         for c in Classes:
#             self._probs.append(respace(probs[:,:,:,c.value], spacing, nspacing, cv2.INTER_LINEAR))

#         self._probs = np.stack(self._probs, axis=-1)
#         self._masks = get_maximal_connected_component(self._probs.argmax(-1).astype(np.uint8), connectivity=18)
#         self._spacing = nspacing[0]

#     @fields(['point', 'normal'])
#     def aortic_iterator(self):
#         yield from self._aortic_iterator()

#     def _aortic_iterator(self):
#         raise NotImplemented()

#     @staticmethod
#     def _get_plane_normals(normal):
#         ex = np.array([0., 1., 0.])
#         ex -= ex.dot(normal) * normal
#         ex /= np.linalg.norm(ex)

#         ey = np.cross(normal, ex)

#         return np.array([ex, ey])

#     @staticmethod
#     def _get_idces(shape, point, normal):
#         eXY = CenterlineDescriptor._get_plane_normals(normal)
#         mXY = shape

#         eZ = np.cross(eXY[0], eXY[1])
#         R = np.array([eXY[0], eXY[1], eZ], dtype=np.float32).T

#         mX, mY = int(mXY[0]), int(mXY[1])
#         Xs = np.arange(-mX/2, +mX/2)
#         Ys = np.arange(-mY/2, +mY/2)
#         PP = np.zeros((3, mX, mY), dtype=np.float32)
#         PP[0, :, :] = Xs.reshape(mX, 1)
#         PP[1, :, :] = Ys.reshape(1, mY)

#         idces = np.einsum('il,ljk->ijk', R, PP) + point.reshape(3, 1, 1)

#         return idces

#     def _imgs_normal_slice(self, point, normal):
#         """ NOTE point transformed to slice center
#         """
#         idces = CenterlineDescriptor._get_idces(self._imgs.shape[1:], point, normal)
#         slice = map_coordinates(self._imgs, idces, order=1, mode='constant', cval=AIR_HU)

#         return slice

#     def _masks_normal_slice(self, point, normal):
#         idces = self._get_idces(self._masks.shape[1:], point, normal)

#         prob_slices = list()

#         for c in Classes:
#             if c.value == Classes.BACKGROUND.value:
#                 value = 1
#             else:
#                 value = 0

#             prob_slice = map_coordinates(self._probs[:,:,:,c.value], idces, order=1, mode='constant', cval=value)
#             prob_slices.append(prob_slice)

#         slice = np.argmax(prob_slices, axis=0)

#         blobs = cc3d.connected_components(slice != Classes.BACKGROUND.value)
#         slice[blobs != blobs[int(self._masks.shape[1]/2), int(self._masks.shape[2]/2)]] = Classes.BACKGROUND.value

#         return slice
