import numpy as np
import nibabel as nib
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math
import nibabel as nib
import numpy as np
from typing import Callable
import nibabel as nib
import numpy as np
import nibabel as nib
import surface_distance

def metrics(epoch, key, idx, alg, phase, config):
    if phase == 'val':
        DIR = config['VAL_OUTPUT']
    else:
        DIR = config['TEST_OUTPUT']
    pred = np.load(DIR + '/pred_centerline_points' +'/' + alg +f'/epoch{epoch}' + '/' +key+'_pred_points.npz')['data']
    true = np.load(DIR + '/true_centerline_points' +'/' + key+'_true_points.npz')['data']

    voxelized_centerline_true = np.zeros((pred[:,0].max(),512,512), dtype=np.uint8)

    for z, x, y in true:
        try:
            voxelized_centerline_true[int(round(z)), int(round(x)), int(round(y))] = 1
        except Exception as e:
            continue

    voxelized_centerline_pred = np.zeros((pred[:,0].max(),512,512), dtype=np.uint8)

    for z, x, y in pred:
        try:
            voxelized_centerline_pred[int(round(z)), int(round(x)), int(round(y))] = 1
        except:
            continue
    distances = surface_distance.compute_surface_distances(
        voxelized_centerline_true.astype(bool),
        voxelized_centerline_pred.astype(bool),
        spacing_mm=(1,1,1)
    )

    sd1mm = surface_distance.compute_surface_dice_at_tolerance(distances, tolerance_mm=1)
    sd3mm = surface_distance.compute_surface_dice_at_tolerance(distances, tolerance_mm=3)

    assd = surface_distance.compute_average_surface_distance(distances)[0]
    hd = surface_distance.compute_robust_hausdorff(distances, percent=100)

    return sd1mm, sd3mm, assd, hd

def linear_frechet(p: np.ndarray, q: np.ndarray, dist_func: Callable[[np.ndarray, np.ndarray], float]) -> float:
    n_p = p.shape[0]
    n_q = q.shape[0]
    ca = np.zeros((n_p, n_q), dtype=np.float64)

    for i in range(n_p):
        for j in range(n_q):
            d = dist_func(p[i], q[j])

            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j],
                                   ca[i - 1, j - 1],
                                   ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = d
    return ca[n_p - 1, n_q - 1]
def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))
def chamfer_distance(arr1, arr2):
    distance_1_to_2 = 0
    distance_2_to_1 = 0

    points1 = arr1
    points2 = arr2

    for p1 in points1:
        distances = np.sqrt(np.sum((points2 - p1)**2, axis=1))
        min_distance = np.min(distances)
        distance_1_to_2 += min_distance

    for p2 in points2:
        distances = np.sqrt(np.sum((points1 - p2)**2, axis=1))
        min_distance = np.min(distances)
        distance_2_to_1 += min_distance

    return (distance_1_to_2 + distance_2_to_1) / (len(arr1) + len(arr2))

def smooth(curve, window_size):
    curve = np.array(curve)
    y_values = curve[:,1]
    x_values = curve[:,0]
    polynomial_order = 2 

    smoothed_y = savgol_filter(y_values, window_size, polynomial_order)
    
    smoothed_curve = np.column_stack((x_values, smoothed_y))
    return smoothed_curve
def divide(bif_arr):
    left_bif = []
    right_bif = []
    for i in range(0, len(bif_arr), 2):
        if bif_arr[i][1] < bif_arr[i+1][1]:
            left_bif.append(bif_arr[i])
            right_bif.append(bif_arr[i+1])
        else:
            left_bif.append(bif_arr[i+1])
            right_bif.append(bif_arr[i])
    return np.array(left_bif), np.array(right_bif)
def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))