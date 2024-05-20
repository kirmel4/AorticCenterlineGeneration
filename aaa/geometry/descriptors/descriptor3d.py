import trimesh
import numpy as np

from aaa.geometry.enum import Classes
from aaa.geometry.tree.tree import Tree
from aaa.geometry.descriptors.descriptor import Descriptor
from aaa.process.post import filter_holes, filter_largest_connected_component
from aaa.geometry.trimesh import get_marching_cubes_surface
from aaa.geometry.vmtk import get_centerline, convert_trimesh_to_vtkmesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vtk import vtkXMLPolyDataWriter

class Descriptor3D(Descriptor):
    def __init__(self, imgs, probs, *, spacing=(1., 1., 1.), nspacing=(1., 1., 1.)):
        Descriptor.__init__(self, imgs, probs, spacing=spacing, nspacing=nspacing)

        self._inner_surface = self.__create_inner_surface()
        self._outer_surface = self.__create_outer_surface()

        self._inner_tree, self._ps = self.__create_inner_tree()

    @staticmethod
    def __filter_masks(masks):
        masks = masks.astype(np.uint8)
        masks = filter_largest_connected_component(masks, connectivity=18)
        masks = filter_holes(masks, ncands=3)

        return masks

    def __create_inner_surface(self):
        return get_marching_cubes_surface(
            self.__filter_masks(
                self._masks == Classes.LUMEN.value
            )
        )

    def __create_outer_surface(self):
        return get_marching_cubes_surface(
            self.__filter_masks(
                self._masks != Classes.BACKGROUND.value
            )
        )

    @staticmethod
    def __create_tree(surface, range_):
        from vtkmodules.util.numpy_support import vtk_to_numpy

        surface = trimesh.intersections.slice_mesh_plane(
            surface,
            [1, 0, 0],
            [range_[0], 0, 0],
            cap=False
        )

        surface = trimesh.intersections.slice_mesh_plane(
            surface,
            [-1, 0, 0],
            [range_[1], 0, 0],
            cap=False
        )

        # surface.export('mesh2.stl')

        vtksurface = convert_trimesh_to_vtkmesh(surface) 
        inner_centerline = get_centerline(vtksurface, length=0.1 / 0.8)
        # print(type(inner_centerline))
        # writer = vtkXMLPolyDataWriter()
        # writer.SetInputData(inner_centerline)
        # writer.SetFileName("centerline2.vtp")
        # writer.Write()
        points = vtk_to_numpy(inner_centerline.GetPoints().GetData())
        points = np.unique(np.array(points), axis=0)
        points = sorted(points, key=lambda x: x[0], reverse=True)

        tree = Tree(points)

        return tree, points

    def __create_inner_tree(self):
        return self.__create_tree(
            self._inner_surface,
            (0, self._masks.shape[0]-1)
        )

    @property
    def inner_surface(self):
        return self._inner_surface

    @property
    def outer_surface(self):
        return self._outer_surface
