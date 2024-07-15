import numpy as np
import nibabel as nib
from scipy import ndimage
import nibabel as nib
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
import torch
import numpy as np
import nibabel as nib
from sklearn.metrics import pairwise_distances

    
def true_centerlines(attraction_field, key, idx, phase, config):
    if phase == 'val':
        DIR = config['VAL_OUTPUT']
    else:
        DIR = config['TEST_OUTPUT']
    centerline = np.zeros(attraction_field.shape[:-1])
    indices = []
    
    for z in tqdm(range(attraction_field.shape[0])):
        for x in range(attraction_field.shape[1]):
            for y in range(attraction_field.shape[2]):
                indices.append(np.add([z, x, y], attraction_field[z, x, y]))

    unique_arrays = set()
    for arr in tqdm(indices):
        unique_arrays.add(tuple(arr))
    unique_arrays = [list(arr) for arr in unique_arrays]
    unique_arrays = np.array(unique_arrays).astype(int)
    for index in unique_arrays:
        try:
            centerline[index[0], index[1], index[2]] += 1
        except:
            continue

    centerline_indices=[]
    biffurcation = 1
    single_arr = []
    for i, layer in enumerate(centerline):
        labels, num_features = ndimage.label(np.where(layer > 0, 1, 0))
        if biffurcation:
            if num_features >= 2:
                max_dict = {feature: np.max(layer * np.where(labels ==feature, 1,0)) for feature in range(1,num_features+1)}
                max_sorted_dict = sorted(max_dict.items(), key=lambda item: item[1], reverse=True)
                two_maxes = [item[0] for item in max_sorted_dict[:3]]
                max_index_masked_first = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[0], 1,0).astype(bool)), layer.shape)
                max_index_masked_second = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[1], 1,0).astype(bool)), layer.shape) 
                if num_features == 3:
                    max_index_masked_first = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[0], 1,0).astype(bool)), layer.shape)
                    max_index_masked_second = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[2], 1,0).astype(bool)), layer.shape)
                dist = np.linalg.norm(np.array(max_index_masked_first) - np.array(max_index_masked_second))
                if dist < 5:
                    biffurcation = 0
                    bif_arr= np.array(centerline_indices).copy()
                    centerline_indices.append((i,*max_index_masked_first))
                    single_arr.append((i, *max_index_masked_first))
                else:
                    centerline_indices.append((i,*max_index_masked_first))
                    centerline_indices.append((i,*max_index_masked_second))
            else:
                biffurcation = 0
                max_index_masked = np.unravel_index(np.argmax(layer), layer.shape)
                centerline_indices.append((i, *max_index_masked))
                single_arr.append((i, *max_index_masked))
        else:
            max_index_masked = np.unravel_index(np.argmax(layer), layer.shape)
            centerline_indices.append((i, *max_index_masked))
            single_arr.append((i, *max_index_masked))

    centerline_masked = np.zeros(centerline.shape)
    for i, index in enumerate(centerline_indices):
            centerline_masked[index[0]][index[1]][index[2]] = 1
    if not os.path.exists(DIR + '/true_centerline_points'):
        os.mkdir(DIR + '/true_centerline_points')
    np.savez(DIR + f'/true_centerline_points/{key}_true_points', data = centerline_indices)

def non_maximum_suppression(pred_attraction, prob_mask, epoch, key, idx, phase, config, radius = 2):
    if phase == 'val':
        DIR = config['VAL_OUTPUT']
    else:
        DIR = config['TEST_OUTPUT']

    pred_magnitude = np.linalg.norm(pred_attraction, axis = 0)
    pred_mask = prob_mask.argmax(axis = 0)

    flattened_arr = np.where(pred_mask ==1 , pred_magnitude, 100000).flatten()

    indices_of_min_values = np.argpartition(flattened_arr, 500)[:500]
    indices_3d = np.unravel_index(indices_of_min_values, pred_magnitude.shape)
    combined_indices = np.vstack(indices_3d).T
    sorted = combined_indices[np.argsort(combined_indices[:, 0])]
    scores  = []
    for x in sorted:
        scores.append(pred_magnitude[x[0], x[1], x[2]])
    scores = np.array(scores)

    sorted_indices = np.argsort(-scores)
    suppressed = np.zeros(len(sorted), dtype=bool)
    selected_points = []

    for i in sorted_indices:
        if suppressed[i]:
            continue

        distances = np.sqrt(np.sum((sorted - sorted[i]) ** 2, axis=1))
        suppression_mask = distances < radius
        suppressed = suppressed | suppression_mask
        suppressed[i] = False
        selected_points.append(sorted[i])

    
    if not os.path.exists(DIR + '/pred_centerline_points'):
        os.mkdir(DIR + '/pred_centerline_points')
    if not os.path.exists(DIR + '/pred_centerline_points'+ '/k_centers'):
        os.mkdir(DIR + '/pred_centerline_points'+ '/k_centers')
    if not os.path.exists(DIR + f'/pred_centerline_points/k_centers/epoch{epoch}'):
        os.mkdir(DIR + f'/pred_centerline_points/k_centers/epoch{epoch}')
    np.savez(DIR + f'/pred_centerline_points/k_centers/epoch{epoch}/{key}_pred_points', data = np.array(selected_points))

def greedy_k_centers(pred_attraction, prob_mask, epoch, key, idx, phase, config, k = 160):
    if phase == 'val':
        DIR = config['VAL_OUTPUT']
    else:
        DIR = config['TEST_OUTPUT']
    pred_magnitude = np.linalg.norm(pred_attraction, axis = 0)
    pred_mask = prob_mask.argmax(axis = 0)

    flattened_arr = np.where(pred_mask ==1 , pred_magnitude, 100000).flatten()
    indices_of_min_values = np.argpartition(flattened_arr, 500)[:500]
    indices_3d = np.unravel_index(indices_of_min_values, pred_magnitude.shape)
    combined_indices = np.vstack(indices_3d).T
    points = combined_indices[np.argsort(combined_indices[:, 0])]

    n = len(points)
    centers = [points[np.random.randint(n)]]

    for _ in range(k - 1):

        distances_to_nearest_center = pairwise_distances(points, centers).min(axis=1)
        farthest = points[np.argmax(distances_to_nearest_center)]
        centers.append(farthest)

    if not os.path.exists(DIR + '/pred_centerline_points'):
        os.mkdir(DIR + '/pred_centerline_points')
    if not os.path.exists(DIR + '/pred_centerline_points'+ '/nms'):
        os.mkdir(DIR + '/pred_centerline_points'+ '/nms')
    if not os.path.exists(DIR + f'/pred_centerline_points/nms/epoch{epoch}'):
        os.mkdir(DIR + f'/pred_centerline_points/nms/epoch{epoch}')
    np.savez(DIR + f'/pred_centerline_points/nms/epoch{epoch}/{key}_pred_points', data = np.array(centers))

def mag_and_field_alg(prob_attraction, prob_masks, epoch, key, idx, phase, config):
    if phase == 'val':
        DIR = config['VAL_OUTPUT']
    else:
        DIR = config['TEST_OUTPUT']
    pred_attraction_field =np.moveaxis(prob_attraction,0, -1)
    pred_centerline = np.zeros(pred_attraction_field.shape[:-1])
    pred_mask = prob_masks.argmax(axis = 0)
    pred_centerline_indices=[]
    biffurcation = 1
    pred_single_arr = []
    pred_bif_arr = []
    pred_att_magnitudes = np.linalg.norm(pred_attraction_field, axis = -1)

    for i, layer in enumerate(pred_att_magnitudes):
        labels, num_features = ndimage.label(pred_mask[i])
        if biffurcation:
            if num_features >= 2:
                min1 = np.unravel_index(np.argmin(layer * np.where(labels ==1, 1,10000000)), layer.shape)
                min2 = np.unravel_index(np.argmin(layer * np.where(labels ==2, 1,10000000)), layer.shape)
                pred_centerline_indices.append(np.array((i,*np.round(pred_attraction_field[(i,*min1)])[1:]+[i,*min1][1:])).astype(int))
                pred_centerline_indices.append(np.array((i,*np.round(pred_attraction_field[(i,*min2)])[1:]+[i,*min2][1:])).astype(int))
            else:
                min_index = np.unravel_index(np.argmin(layer*np.where(labels ==1, 1,10000000), axis=None), layer.shape)
                layer[min_index] = 10000000
                second_min_index = np.unravel_index(np.argmin(layer*np.where(labels ==1, 1,10000000), axis=None), layer.shape)
                dist = np.linalg.norm(np.array(min_index) - np.array(second_min_index))
                if dist < 10:
                    pred_bif_arr = np.array(pred_centerline_indices).copy()
                    pred_centerline_indices.append(np.array((i,*np.round(pred_attraction_field[(i,*min_index)])[1:]+[i,*min_index][1:])).astype(int))
                    pred_single_arr.append(np.array((i,*np.round(pred_attraction_field[(i,*min_index)])[1:]+[i,*min_index][1:])).astype(int))
                    biffurcation = 0
                else:
                    pred_centerline_indices.append(np.array((i,*np.round(pred_attraction_field[(i,*min_index)])[1:]+[i,*min_index][1:])).astype(int))
                    pred_centerline_indices.append(np.array((i,*np.round(pred_attraction_field[(i,*second_min_index)])[1:]+[i,*second_min_index][1:])).astype(int))

        else:
            min_index = np.unravel_index(np.argmin(layer*np.where(labels ==1, 1,100000), axis=None), layer.shape)
            pred_centerline_indices.append(np.array((i,*np.round(pred_attraction_field[(i,*min_index)])[1:]+[i,*min_index][1:])).astype(int))
            pred_single_arr.append(np.array((i,*np.round(pred_attraction_field[(i,*min_index)])[1:]+[i,*min_index][1:])).astype(int))

    pred_centerline_masked = np.zeros(pred_centerline.shape)
    for i, index in enumerate(pred_centerline_indices):
        try:
            pred_centerline_masked[index[0]][index[1]][index[2]] = 1
        except:
            continue

    

    if not os.path.exists(DIR + '/pred_centerline_points'):
        os.mkdir(DIR + '/pred_centerline_points')
    if not os.path.exists(DIR + '/pred_centerline_points'+ '/mag_and_field'):
        os.mkdir(DIR + '/pred_centerline_points'+ '/mag_and_field')
    if not os.path.exists(DIR + f'/pred_centerline_points/mag_and_field/epoch{epoch}'):
        os.mkdir(DIR + f'/pred_centerline_points/mag_and_field/epoch{epoch}')
    np.savez(DIR + f'/pred_centerline_points/mag_and_field/epoch{epoch}/{key}_pred_points', data = pred_centerline_indices)

def mag(prob_attraction, prob_masks, epoch, idx, phase, config):
    if phase == 'val':
        DIR = config['VAL_OUTPUT']
    else:
        DIR = config['TEST_OUTPUT']
    pred_attraction_field =np.moveaxis(prob_attraction,0, -1)
    pred_centerline = np.zeros(pred_attraction_field.shape[:-1])
    pred_mask = prob_masks.argmax(axis = 0)
    pred_centerline_indices=[]
    biffurcation = 1
    pred_single_arr = []
    pred_bif_arr = []
    pred_att_magnitudes = np.linalg.norm(pred_attraction_field, axis = -1)

    for i, layer in enumerate(pred_att_magnitudes):
        labels, num_features = ndimage.label(pred_mask[i])
        if biffurcation:
            if num_features >= 2:

                min1 = np.unravel_index(np.argmin(layer * np.where(labels ==1, 1,10000000)), layer.shape)
                min2 = np.unravel_index(np.argmin(layer * np.where(labels ==2, 1,10000000)), layer.shape)

                pred_centerline_indices.append((i, *min1))
                pred_centerline_indices.append((i, *min2))
            else:
                min_index = np.unravel_index(np.argmin(layer*np.where(labels ==1, 1,10000000), axis=None), layer.shape)
                layer[min_index] = 10000000
                second_min_index = np.unravel_index(np.argmin(layer*np.where(labels ==1, 1,10000000), axis=None), layer.shape)
                dist = np.linalg.norm(np.array(min_index) - np.array(second_min_index))
                if dist < 10:
                    pred_bif_arr = np.array(pred_centerline_indices).copy()
                    pred_centerline_indices.append((i,*min_index))
                    pred_single_arr.append((i,*min_index))
                    biffurcation = 0
                else:
                    pred_centerline_indices.append((i, *min_index))
                    pred_centerline_indices.append((i, *second_min_index))

        else:
            min_index = np.unravel_index(np.argmin(layer*np.where(labels ==1, 1,100000), axis=None), layer.shape)
            pred_centerline_indices.append((i, *min_index))
            pred_single_arr.append((i, *min_index))

    pred_centerline_masked = np.zeros(pred_centerline.shape)
    for i, index in enumerate(pred_centerline_indices):
        pred_centerline_masked[index[0]][index[1]][index[2]] = 1

    if not os.path.exists(DIR + '/pred_centerline_points'):
        os.mkdir(DIR + '/pred__centerline_points')
    if not os.path.exists(DIR + '/pred_centerline_points'+ '/mag'):
        os.mkdir(DIR + '/pred_centerline_points'+ '/mag')
    if not os.path.exists(DIR + f'/pred_centerline_points/mag/epoch{epoch}'):
        os.mkdir(DIR + f'/pred_centerline_points/mag/epoch{epoch}')
    np.savez(DIR + f'/pred_centerline_points/mag/epoch{epoch}/pred_single_arr{idx}', data = pred_single_arr)
    np.savez(DIR + f'/pred_centerline_points/mag/epoch{epoch}/pred_bif_arr{idx}', data = pred_bif_arr)

def field(prob_attraction, prob_masks, epoch, idx, phase, config):
    if phase == 'val':
        DIR = config['VAL_OUTPUT']
    else:
        DIR = config['TEST_OUTPUT']
    pred_attraction_field =np.moveaxis(prob_attraction,0, -1)
    pred_centerline = np.zeros(pred_attraction_field.shape[:-1])
    pred_mask = prob_masks.argmax(axis = 0)
    pred_centerline_indices=[]
    biffurcation = 1
    pred_single_arr = []
    pred_bif_arr = []
    pred_mask_stacked = np.stack((pred_mask, pred_mask, pred_mask), axis = 0)
    pred_attraction_field = prob_attraction * pred_mask_stacked
    pred_attraction_field =np.moveaxis(prob_attraction,0, -1)

    pred_centerline = np.zeros(pred_attraction_field.shape[:-1])
    pred_indices = []

    for z in tqdm(range(pred_attraction_field.shape[0])):
        for x in range(pred_attraction_field.shape[1]):
            for y in range(pred_attraction_field.shape[2]):
                pred_indices.append(np.round(np.add([z, x, y], pred_attraction_field[z, x, y])))
    pred_indices = np.array(pred_indices).astype(int)

    for index in tqdm(pred_indices):
        try:
            pred_centerline[index[0], index[1], index[2]] +=1
        except:
            continue

    pred_centerline_indices=[]
    biffurcation = 1
    pred_single_arr = []
    pred_bif_arr = []
    
    for i, layer in enumerate(pred_centerline):
                labels, num_features = ndimage.label(pred_mask[i])
                if biffurcation:
                    if num_features >= 2:
                        max_dict = {feature: np.max(layer * np.where(labels ==feature, 1,0)) for feature in range(1,num_features+1)}
                        max_sorted_dict = sorted(max_dict.items(), key=lambda item: item[1], reverse=True)
                        two_maxes = [item[0] for item in max_sorted_dict if item[1] != 0]
                        if len(two_maxes) == 1:
                            max_index = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[0], 1,0).astype(bool)), layer.shape)

                            layer[max_index] = 0
                            second_max_index = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[0], 1,0).astype(bool)), layer.shape)
                            dist = np.linalg.norm(np.array(max_index) - np.array(second_max_index))
                            if dist < 7:
                                pred_bif_arr = np.array(pred_centerline_indices).copy()
                                pred_centerline_indices.append((i,*max_index))
                                pred_single_arr.append((i,*max_index))
                                biffurcation = 0
                            else:
                                pred_centerline_indices.append((i, *max_index))
                                pred_centerline_indices.append((i, *second_max_index))
                        else:
                            max_index_masked_first = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[0], 1,0).astype(bool)), layer.shape)
                            max_index_masked_second = np.unravel_index(np.argmax(layer * np.where(labels ==two_maxes[1], 1,0).astype(bool)), layer.shape)
                            pred_centerline_indices.append((i, *max_index_masked_first ))
                            pred_centerline_indices.append((i, *max_index_masked_second ))
                    else:
                        max_index = np.unravel_index(np.argmax(layer*(pred_mask[i].astype(bool)), axis=None), layer.shape)
                        layer[max_index] = 0
                        second_max_index = np.unravel_index(np.argmax(layer*(pred_mask[i].astype(bool)), axis=None), layer.shape)
                        dist = np.linalg.norm(np.array(max_index) - np.array(second_max_index))
                        if dist < 7 or second_max_index == (0, 0):
                            pred_bif_arr = np.array(pred_centerline_indices).copy()
                            pred_centerline_indices.append((i,*max_index))
                            pred_single_arr.append((i,*max_index))
                            biffurcation = 0
                        else:
                            pred_centerline_indices.append((i, *max_index))
                            pred_centerline_indices.append((i, *second_max_index))

                else:
                    max_index_masked_first = np.unravel_index(np.argmax(layer*pred_mask[i].astype(bool)), layer.shape)
                    pred_centerline_indices.append((i, *max_index_masked_first))
                    pred_single_arr.append((i, *max_index_masked_first))

    pred_centerline_masked = np.zeros(pred_centerline.shape)
    for i, index in enumerate(pred_centerline_indices):
            pred_centerline_masked[index[0]][index[1]][index[2]] = 1

    if not os.path.exists(DIR + '/pred_centerline_points'):
        os.mkdir(DIR + '/pred__centerline_points')
    if not os.path.exists(DIR + '/pred_centerline_points'+ '/field'):
        os.mkdir(DIR + '/pred_centerline_points'+ '/field')
    if not os.path.exists(DIR + f'/pred_centerline_points/field/epoch{epoch}'):
        os.mkdir(DIR + f'/pred_centerline_points/field/epoch{epoch}')
    np.savez(DIR + f'/pred_centerline_points/field/epoch{epoch}/pred_single_arr{idx}', data = pred_single_arr)
    np.savez(DIR + f'/pred_centerline_points/field/epoch{epoch}/pred_bif_arr{idx}', data = pred_bif_arr)

 
