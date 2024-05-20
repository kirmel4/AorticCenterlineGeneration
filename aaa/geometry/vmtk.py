import vtk
import numpy as np

from vmtk import vtkvmtk, vmtklineresampling
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray

def convert_trimesh_to_vtkmesh(trimesh):
    vtkmesh = vtk.vtkPolyData()

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.array(trimesh.vertices), deep=1))
    vtkmesh.SetPoints(points)

    cells = vtk.vtkCellArray()
    isize = vtk.vtkIdTypeArray().GetDataTypeSize()
    req_dtype = np.int32 if isize == 4 else np.int64
    n_elems = np.array(trimesh.faces).shape[0]
    n_dim = np.array(trimesh.faces).shape[1]
    cells.SetCells(n_elems,
                   numpy_to_vtkIdTypeArray(
                       np.hstack((np.ones(n_elems)[:, None] * n_dim,
                                  np.array(trimesh.faces))).astype(req_dtype).ravel(),
                       deep=1))
    vtkmesh.SetPolys(cells)

    return vtkmesh

def __wrap_volume_to_vtk(volume):
    array = volume.transpose((2, 1, 0)).ravel()

    vtkArray = numpy_to_vtk( num_array=array,
                             deep=True,
                             array_type=vtk.VTK_UNSIGNED_CHAR )

    vtkImage = vtk.vtkImageData()
    vtkImage.SetDimensions(volume.shape)
    vtkImage.GetPointData().SetScalars(vtkArray)

    return vtkImage

def __extract_sufrace(vtkVolume):
    marchingCubes = vtk.vtkMarchingCubes()
    marchingCubes.SetInputData(vtkVolume)
    marchingCubes.SetValue(0, 1.)
    marchingCubes.Update()

    vtkSurface = marchingCubes.GetOutput()

    return vtkSurface

def __triangulate_surface(vtkSurface):
    surfaceTriangulator = vtk.vtkTriangleFilter()
    surfaceTriangulator.SetInputData(vtkSurface)
    surfaceTriangulator.PassLinesOff()
    surfaceTriangulator.PassVertsOff()
    surfaceTriangulator.Update()

    vtkSurface = surfaceTriangulator.GetOutput()

    return vtkSurface

def __clean_surface(vtkSurface):
    surfaceCleaner = vtk.vtkCleanPolyData()
    surfaceCleaner.SetInputData(vtkSurface)
    surfaceCleaner.Update()

    vtkSurface = surfaceCleaner.GetOutput()

    return vtkSurface

def __smooth_surface( vtkSurface,
                      niterations=20,
                      pass_band=1e-4,
                      edge_angle=180,
                      is_boundary_smoothing=True,
                      is_feature_edge_smoothing=False ):
    polyDataFilter = vtk.vtkWindowedSincPolyDataFilter()
    polyDataFilter.SetInputData(vtkSurface)
    polyDataFilter.SetNumberOfIterations(niterations)
    polyDataFilter.SetPassBand(pass_band)
    polyDataFilter.SetEdgeAngle(edge_angle)
    polyDataFilter.SetBoundarySmoothing(is_boundary_smoothing)
    polyDataFilter.SetFeatureEdgeSmoothing(is_feature_edge_smoothing)
    polyDataFilter.NonManifoldSmoothingOn()
    polyDataFilter.NormalizeCoordinatesOn()
    polyDataFilter.Update()

    vtkSurface = polyDataFilter.GetOutput()

    return vtkSurface

def __remesh_surface(vtkSurface):
    surfaceRemesher = vtkvmtk.vtkvmtkPolyDataSurfaceRemeshing()
    surfaceRemesher.SetInputData(vtkSurface)
    surfaceRemesher.NumberOfIterations = 5
    surfaceRemesher.TargetEdgeLengthFactor = 0.5
    surfaceRemesher.Update()

    vtkSurface = surfaceRemesher.GetOutput()

    return vtkSurface

def get_marching_cubes_surface(masks):
    vtkVolume = __wrap_volume_to_vtk(masks)
    vtkSurface = __extract_sufrace(vtkVolume)
    vtkSurface = __clean_surface(vtkSurface)
    vtkSurface = __triangulate_surface(vtkSurface)

    return vtkSurface

def get_smoothed_marching_cubes_surface(masks):
    vtkSurface = get_marching_cubes_surface(masks)
    vtkSurface = __smooth_surface(vtkSurface)
    #vtkSurface = __remesh_surface(vtkSurface)
    vtkSurface = __clean_surface(vtkSurface)

    return vtkSurface

def __extract_network(vtkSurface):
    networkExtraction = vtkvmtk.vtkvmtkPolyDataNetworkExtraction()
    networkExtraction.SetInputData(vtkSurface)
    networkExtraction.SetAdvancementRatio( 1.1 )
    networkExtraction.Update()

    vtkPolydata = networkExtraction.GetOutput()

    return vtkPolydata

def __resample_polydata(vtkPolydata, length=1.):
    lineResampling = vmtklineresampling.vmtkLineResampling()
    lineResampling.Surface = vtkPolydata
    lineResampling.Length = length
    lineResampling.Execute()

    vtkPolydata = lineResampling.Surface

    return vtkPolydata

def get_centerline(vtkSurface, length=1.):
    vtkPolyData = __extract_network(vtkSurface)
    vtkPolyData = __resample_polydata(vtkPolyData, length=length)

    return vtkPolyData
