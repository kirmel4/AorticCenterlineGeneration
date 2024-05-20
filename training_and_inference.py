import nibabel as nib
import numpy as np
import nibabel as nib
import glob
import torch
import random
import os
from torch import Tensor
from collections import OrderedDict
from torch.utils.data import Dataset
from scipy import ndimage
from aaa.utils import torch_float, torch_long, io
from scipy.spatial.distance import cdist
from aaa.datasets.datasetGz import aaaIGzDataset, aaaIMGzDataset
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import math
import torch.nn.functional as F
from typing import Callable
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from frechetdist import frdist
from torchmetrics.functional import dice
from aaa.metrics import dice_score, sensitivity_score, volumetric_similarity_score
from aaa.models.unet3d_sm import Unet3dSm
from aaa.datasets.datasetGz import aaaIGzDataset, aaaIMGzDataset
from aaa.geometry.descriptors.descriptor import Descriptor
from aaa.geometry.descriptors.descriptor3d import Descriptor3D
from aaa.models.my_unet import my_unet
import nibabel as nib
import numpy as np
import nibabel as nib
import glob
import torch
from torch import nn
import matplotlib.pyplot as plt
from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
from aaa.geometry.enum import Classes
import random
import os
from pathlib import Path
from scipy.spatial import KDTree
from tqdm import tqdm
from aaa.datasets.MyDataset import SegAttDataset
from aaa.utils import ( init_determenistic, init_logging,
                            torch_float, torch_long,
                            load_yaml_config, unchain )
from aaa.losses.threedim import dice_with_logits_loss, focal_with_logits_loss
from aaa.process.pre import ( aug_parallelization, aug_uda,
                                  apply_voxel_augmentation,
                                  apply_voxel_augmentations, apply_only_voxel_augmentations,
                                  apply_image_augmentations, apply_only_image_augmentations,
                                  voxel_random_selector, voxel_sequential_selector, voxel_batch_generator,
                                  VoxelRandomSampler, VoxelSequentialSampler )
from torch.utils.data import Dataset, DataLoader
from aaa.inference.threedim import inference_default, inference_with_vertical_flip
from collections import OrderedDict, defaultdict
from aaa.utils import config as cfg
import copy
import json
import time
import click
from clearml import Task
import torch
import numpy as np
import nibabel as nib
import os
from torch import Tensor
from collections import OrderedDict
from torch.utils.data import Dataset
from aaa.utils import torch_float, torch_long, io
from aaa.datasets.datasetGz import aaaIGzDataset, aaaIMGzDataset
from torchmetrics.functional import dice

DIR = './'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device}")

config = dict()

debugging_info = { 'epoch_loss_trace': list(),
                   'epoch_score_trace': list(),
                   'epoch_additional_score_trace': list(),
                   'epoch_times': list(),
                   'max_memory_consumption': 0.,
                 }

def init_global_config(**kwargs):
    cfg.init_timestamp(config)
    cfg.init_run_command(config)
    cfg.init_kwargs(config, kwargs)
    init_logging(config, __name__, config['LOGGER_TYPE'], filename=config['PREFIX']+'logger_name.txt')
    cfg.init_device(config)
    cfg.init_verboser(config, logger=config['LOGGER'])
    cfg.init_options(config)

class SegAttDataset(aaaIGzDataset):
    #data in z-x-y format
    def __init__(self, imgs, masks, attraction_fields, channels=None):
        self.channels = channels

        self.imgs = imgs
        self.masks = masks
        self.attraction_fields = attraction_fields

        self.keys = list()
        self.shapes = list()

        for key in self.imgs:
            mask = self.masks[key]

            self.keys.append(key)
            self.shapes.append(mask.shape[:-1]) ###
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        from aaa.augmentations.unet3d_sm import train_aug
        key, selector = idx
        voxel = self.imgs[key][selector]
        mask = self.masks[key][selector]
        attraction_field = self.attraction_fields[key][selector]
        voxel = io.split_images(voxel[:, :, None], config['LOADER_OPTIONS']) # :

        return voxel, mask, attraction_field, key, selector

def collate_fn(batch):
        voxels, masks, attraction_fields, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = torch_float(voxels, torch.device('cpu'))
        masks = torch_float(masks, torch.device('cpu'))
        attraction_fields = torch_float(attraction_fields, torch.device('cpu'))

        return voxels, masks, attraction_fields, selections

def load_data_with_labels(data):
    
    datapath = config['DATAPATH'] 

    imgs = OrderedDict()
    masks = { }
    attraction_fields = {}

    for name, keys in config['SPLIT_OPTIONS'].items():
        for key in keys:
            image = np.load(datapath + '/' + key + '/' + key + '_img.npz')['data']
            
            imgs[key] = image #####
 

            mask = np.load(datapath + '/' + key + '/' + key + '_segmentation.npz')['data']
            masks[key] = mask

            attraction_field = np.load(datapath + '/' + key + '/' + key + '_attraction.npz')['data']
            attraction_fields[key] = attraction_field
        data[name] = SegAttDataset(
            { key: imgs[key] for key in keys },
            { key: masks[key] for key in keys },
            { key: attraction_fields[key] for key in keys}, channels = config['LOADER_OPTIONS']['channels']
        )

def load_data():
    data = { }

    load_data_with_labels(data)

    return data

def inner_supervised(model, voxels_batch, masks_batch, attraction_field_batch, step_idx, epoch):


    voxels_batch = voxels_batch.to(device)
    masks_batch = masks_batch.to(device)
    attraction_field_batch = attraction_field_batch.to(device)
    voxels_batch = torch.moveaxis(voxels_batch, 3, 1)
    pred_mask_batch, pred_attraction_batch = model(voxels_batch)

    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    masks_batch = masks_batch[:,:,:,:,1]

    loss_mask = dice_with_logits_loss(masks_batch, pred_mask_batch, average='binary', activation='softmax') +\
                focal_with_logits_loss(masks_batch.long(), pred_mask_batch)
    masks_batch_stacked = torch.moveaxis(torch.stack((masks_batch, masks_batch, masks_batch)),0,1)
    pred_mask_batch_stacked = torch.moveaxis(torch.stack((pred_mask_batch.argmax(axis = 1), pred_mask_batch.argmax(axis = 1), pred_mask_batch.argmax(axis = 1))),0,1)

    mse = mse_loss(torch.moveaxis(attraction_field_batch,-1,1)*masks_batch_stacked, pred_attraction_batch*pred_mask_batch_stacked)*25 

    att_magnitudes = torch.linalg.norm(attraction_field_batch, dim = -1)
    pred_att_magnitudes = torch.linalg.norm(pred_attraction_batch, dim = 1)

    mag_lumen = att_magnitudes*masks_batch
    pred_mag_lumen = pred_att_magnitudes*pred_mask_batch.argmax(axis = 1)

    vector_length_regularization = torch.mean(torch.abs(mag_lumen - pred_mag_lumen))

    regularization_coefficient = 125
    vector_length_regularization *= regularization_coefficient

    loss_attraction = (mse+vector_length_regularization)*1
    print(loss_mask, loss_attraction)

    return loss_mask, loss_attraction, mse, vector_length_regularization

def inner_train_loop(model, opt, dataset, epoch):
    model.train()

    batch_mask_losses = list()
    batch_att_losses = list()
    batch_mse_losses = list()
    batch_reg_losses = list()

    datasampler = VoxelRandomSampler( config['VOXELING_OPTIONS']['voxel_shape'],
                                      dataset.keys,
                                      dataset.shapes,
                                      config['N_ITERATIONS'] * config['BATCH_SIZE'])

    dataloader = DataLoader( dataset,
                             batch_size= config['BATCH_SIZE'],
                             sampler=datasampler,
                             collate_fn=collate_fn,
                             num_workers=config['NJOBS'],
                             pin_memory=False,
                             prefetch_factor=1 )

    opt.zero_grad()

    for step_idx, (voxels_batch, masks_batch, attraction_field_batch, _) in config['VERBOSER'](enumerate(dataloader), total=len(dataloader)):
        print(step_idx)
  
        loss_mask, loss_attraction, mse, reg = inner_supervised(model, voxels_batch, masks_batch, attraction_field_batch, step_idx, epoch)
        loss_attraction_normalized = loss_attraction
        loss_mask = 100*loss_mask
        loss = loss_attraction_normalized+loss_mask

        loss.backward()
        opt.step()
        opt.zero_grad()
    
        batch_mask_losses.append(loss_mask.item())
        batch_att_losses.append(loss_attraction_normalized.item())
        batch_mse_losses.append(mse.item())
        batch_reg_losses.append(reg.item())

    return np.mean(batch_mask_losses), np.mean(batch_att_losses), np.mean(batch_mse_losses), np.mean(batch_reg_losses)

def inner_val_loop(model, dataset, epoch):
    model.eval()
    save_pred_mask=[]
    save_mask = []
    save_attraction = []
    save_pred_attraction = []
    save_image = []
    centerlines = []
    pred_centerlines = []
    frechet_bif = []
    frechet_single = []
    chamfer = []
    frechet = []
    chamfer_bif = []
    chamfer_single = []

    for idx in config['VERBOSER'](np.arange(len(dataset)), total=len(dataset)):

        key = dataset.keys[idx]
        shape = dataset.shapes[idx]
        save_mask

        datasampler = VoxelSequentialSampler( config['VOXELING_OPTIONS']['voxel_shape'],
                                              [key],
                                               [shape], 
                                              config['VOXELING_OPTIONS']['steps'] )

        dataloader = DataLoader( dataset,
                                 batch_size=config['BATCH_SIZE'], 
                                 sampler=datasampler,
                                 collate_fn=collate_fn,
                                 num_workers=config['NJOBS'],
                                 pin_memory=False,
                                 prefetch_factor=1 )

        imgs = dataset.imgs[key]
        masks = dataset.masks[key].astype(int)
        attraction_fields = dataset.attraction_fields[key]
        save_mask.append(masks[:130,:,:,:])
        save_attraction.append(attraction_fields[:130,:,:,:])
        save_image.append(imgs[:130, :, :])
        prob_masks = np.zeros((config['N_CLASSES'], *shape))
        prob_attraction = np.zeros((3, *shape))
        for voxels_batch, _, __, selections in tqdm(dataloader):
            with torch.no_grad():
                voxels_batch = voxels_batch.to(config['DEVICE'])
                voxels_batch = torch.moveaxis(voxels_batch, 3, 1)
                pred_mask_batch, pred_attraction_batch = model(voxels_batch)
                prob_attraction[(0,*selections[-1][-1])] += pred_attraction_batch[0][0].cpu().data.numpy()
                prob_attraction[(1,*selections[-1][-1])] += pred_attraction_batch[0][1].cpu().data.numpy()
                prob_attraction[(2,*selections[-1][-1])] += pred_attraction_batch[0][2].cpu().data.numpy()
                prob_masks[(0,*selections[-1][-1])] += pred_mask_batch[0][0].cpu().data.numpy()
                prob_masks[(1,*selections[-1][-1])] += pred_mask_batch[0][1].cpu().data.numpy()
                
        save_pred_mask.append(prob_masks[:,:130,:,:])
        save_pred_attraction.append(prob_attraction[:,:130,:,:])
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
                    if dist < 7:
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

        attraction_field =attraction_fields
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
            centerline[index[0], index[1], index[2]] += 1

        centerline_indices=[]
        biffurcation = 1
        single_arr = []
        bif_arr = []
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
                    bif_arr= np.array(centerline_indices).copy()
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

        centerlines.append(centerline_masked[:130,:,:])
        pred_centerlines.append(pred_centerline_masked[:130,:,:])
        if not os.path.exists(DIR + 'testarrays'):
            os.mkdir(DIR + 'testarrays')
        np.savez(DIR + f'testarrays/single_arr{epoch}_{idx}', data = single_arr)
        np.savez(DIR + f'testarrays/pred_single_arr{epoch}_{idx}', data = pred_single_arr)
        np.savez(DIR + f'testarrays/bif_arr{epoch}_{idx}', data = bif_arr)
        np.savez(DIR + f'testarrays/pred_bif_arr{epoch}_{idx}', data = pred_bif_arr)

        pred_left_bif, pred_right_bif = divide(pred_bif_arr)
        single_arr = np.array(single_arr)
        pred_single_arr = np.array(pred_single_arr)
        window_size = 20
        try:
            z_points_left = np.concatenate((pred_left_bif[:,0], pred_single_arr[:,0]))
            y_points_left= np.concatenate((pred_left_bif[:,2], pred_single_arr[:,2]))

            spline_interp = interp1d(z_points_left, y_points_left, kind='cubic')

            z_interp_left = np.linspace(min(z_points_left), max(z_points_left), 200)
            y_interp_left = spline_interp(z_interp_left)
            interp_sm_leftzy = smooth(np.stack((z_interp_left, y_interp_left), axis = 1), window_size)

            z_points_right = np.concatenate((pred_right_bif[:,0], pred_single_arr[:,0]))
            y_points_right= np.concatenate((pred_right_bif[:,2], pred_single_arr[:,2]))

            spline_interp = interp1d(z_points_right, y_points_right, kind='cubic')

            z_interp_right= np.linspace(min(z_points_right), max(z_points_right), 200)
            y_interp_right = spline_interp(z_interp_right)
            interp_sm_rightzy= smooth(np.stack((z_interp_right, y_interp_right), axis = 1), window_size)

            x_points_left= np.concatenate((pred_left_bif[:,1], pred_single_arr[:,1]))

            spline_interp = interp1d(z_points_left, x_points_left, kind='cubic')

            x_interp_left = spline_interp(z_interp_left)
            interp_sm_leftzx = smooth(np.stack((z_interp_left, x_interp_left), axis = 1), window_size)

            x_points_right= np.concatenate((pred_right_bif[:,1], pred_single_arr[:,1]))

            spline_interp = interp1d(z_points_right, x_points_right, kind='cubic')

            x_interp_right = spline_interp(z_interp_right)
            interp_sm_rightzx= smooth(np.stack((z_interp_right, x_interp_right), axis = 1), window_size)
            interp_left = []
            interp_right = []
            for i, arr in enumerate(interp_sm_leftzx):
                interp_left.append([arr[0],arr[1], interp_sm_leftzy[i,1]] )
            for i, arr in enumerate(interp_sm_rightzx):
                interp_right.append([arr[0],arr[1], interp_sm_rightzy[i,1]] )
            interp_left = np.array(interp_left)
            interp_right = np.array(interp_right)
        except Exception as e:
            print(str(e))

        if len(pred_single_arr) > len(single_arr):
            pred_single_arr = pred_single_arr[len(pred_single_arr) - len(single_arr):]
        else:
            pred_single_arr = pred_single_arr[len(single_arr) - len(pred_single_arr):]
        try:
            new_bif_arr = []
            for arr in bif_arr:
                if arr[0] <= np.max(pred_bif_arr[:,0]):
                    new_bif_arr.append(arr)
            left_bif, right_bif = divide(new_bif_arr)
        except Exception as e:
            print(str(e))
            continue


        try:
            spline_interp_left_bif = interp1d(pred_left_bif[:,0], pred_left_bif[:,2], kind='cubic')
            z_interp_left_bif = np.linspace(min(pred_left_bif[:,0]), max(pred_left_bif[:,0]), len(left_bif))
            y_interp_left_bif = spline_interp_left_bif(z_interp_left_bif)
            interp_sm_left_bifzy = smooth(np.stack((z_interp_left_bif, y_interp_left_bif), axis = 1), window_size)

            spline_interp_right_bif = interp1d(pred_right_bif[:,0], pred_right_bif[:,2], kind='cubic')
            z_interp_right_bif = np.linspace(min(pred_right_bif[:,0]), max(pred_right_bif[:,0]), len(right_bif))
            y_interp_right_bif = spline_interp_right_bif(z_interp_right_bif)
            interp_sm_right_bifzy = smooth(np.stack((z_interp_right_bif, y_interp_right_bif), axis = 1), window_size)

            spline_interp_single = interp1d(pred_single_arr[:,0], pred_single_arr[:,2], kind='cubic')
            z_interp_single= np.linspace(min(pred_single_arr[:,0]), max(pred_single_arr[:,0]), len(single_arr))
            y_interp_single = spline_interp_single(z_interp_single)
            interp_sm_singlezy = smooth(np.stack((z_interp_single, y_interp_single), axis = 1), window_size)

            spline_interp_left_bif = interp1d(pred_left_bif[:,0], pred_left_bif[:,1], kind='cubic')
            z_interp_left_bif = np.linspace(min(pred_left_bif[:,0]), max(pred_left_bif[:,0]), len(left_bif))
            x_interp_left_bif = spline_interp_left_bif(z_interp_left_bif)
            interp_sm_left_bifzx = smooth(np.stack((z_interp_left_bif, x_interp_left_bif), axis = 1), window_size)

            spline_interp_right_bif = interp1d(pred_right_bif[:,0], pred_right_bif[:,1], kind='cubic')
            z_interp_right_bif = np.linspace(min(pred_right_bif[:,0]), max(pred_right_bif[:,0]), len(right_bif))
            x_interp_right_bif = spline_interp_right_bif(z_interp_right_bif)
            interp_sm_right_bifzx = smooth(np.stack((z_interp_right_bif, x_interp_right_bif), axis = 1), window_size)

            spline_interp_single = interp1d(pred_single_arr[:,0], pred_single_arr[:,1], kind='cubic')
            z_interp_single= np.linspace(min(pred_single_arr[:,0]), max(pred_single_arr[:,0]), len(single_arr))
            x_interp_single = spline_interp_single(z_interp_single)
            interp_sm_singlezx = smooth(np.stack((z_interp_single, x_interp_single), axis = 1), window_size)

            final_left_bif = []
            final_right_bif = []
            final_single = []
            for i, arr in enumerate(interp_sm_left_bifzx):
                final_left_bif.append([arr[0],arr[1], interp_sm_left_bifzy[i,1]])
            for i, arr in enumerate(interp_sm_right_bifzx):
                final_right_bif.append([arr[0],arr[1], interp_sm_right_bifzy[i,1]])
            for i, arr in enumerate(interp_sm_singlezx):
                final_single.append([arr[0],arr[1], interp_sm_singlezy[i,1]])
        except Exception as e:
            print(str(e))
        try:
            frechet_dist_left_bif = linear_frechet(left_bif[:,:2], pred_left_bif[:,:2], euclidean)
            frechet_dist_right_bif = linear_frechet(right_bif[:,:2], pred_right_bif[:,:2], euclidean)
            frechet_dist_single = linear_frechet(single_arr[:,:2], pred_single_arr[:,:2], euclidean)

            chamfer_dist_left_bif = chamfer_distance(left_bif, pred_left_bif)
            chamfer_dist_right_bif = chamfer_distance(right_bif, pred_right_bif)
            chamfer_dist_single = chamfer_distance(single_arr, pred_single_arr)
        except Exception as e:
            print(str(e))
            continue

        #if compare with smoothed

        # frechet_dist_left_bif = linear_frechet(left_bif[:,:2], interp_sm_left_bifzx, euclidean)
        # frechet_dist_right_bif = linear_frechet(right_bif[:,:2], interp_sm_right_bifzx, euclidean)
        # frechet_dist_single = linear_frechet(single_arr[:,:2], interp_sm_singlezx, euclidean)

        # chamfer_dist_left_bif = chamfer_distance(left_bif, final_left_bif)
        # chamfer_dist_right_bif = chamfer_distance(right_bif, final_right_bif)
        # chamfer_dist_single = chamfer_distance(single_arr, final_single)

        chamfer.append(np.mean([chamfer_dist_left_bif, chamfer_dist_right_bif, chamfer_dist_single]))
        frechet.append(np.mean([frechet_dist_left_bif, frechet_dist_right_bif, frechet_dist_single]))

        chamfer_bif.append(np.mean([chamfer_dist_left_bif, chamfer_dist_right_bif]))
        frechet_bif.append(np.mean([frechet_dist_left_bif, frechet_dist_right_bif]))

        chamfer_single.append(chamfer_dist_single)
        frechet_single.append(frechet_dist_single)
    

    try:
        if not os.path.exists(DIR + 'testtraining'):
            os.mkdir(DIR + 'testtraining')
        np.savez_compressed(DIR + f'testtraining/save_images{epoch}', data = np.stack(save_image[:5]))
        
        np.savez_compressed(DIR + f'testtraining/save_pred_masks{epoch}', data =np.stack(save_pred_mask[:5]))
        np.savez_compressed(DIR + f'testtraining/save_masks{epoch}', data =np.stack(save_mask[:5]))

        np.savez_compressed(DIR + f'testtraining/save_pred_attractions{epoch}', data =np.stack(save_pred_attraction[:5]))
        np.savez_compressed(DIR + f'testtraining/save_attractions{epoch}', data = np.stack(save_attraction[:5]))
        np.savez_compressed(DIR + f'testtraining/save_centerlines{epoch}', data = np.stack(centerlines[:5]))
        np.savez_compressed(DIR + f'testtraining/save_pred_centerlines{epoch}', data = np.stack(pred_centerlines[:5]))
    except:
        pass
   
    return np.mean(frechet), np.mean(chamfer),  np.mean(frechet_bif), np.mean(chamfer_bif),  np.mean(frechet_single), np.mean(chamfer_single), 

def fit(model, data):

    model.to(config['DEVICE'])
    opt = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], eps=1e-8)

    epochs_without_going_up = 0
    best_score = 9999
    best_state = copy.deepcopy(model.state_dict())
    loss_mask_arr= []
    loss_att_arr = []
    loss_mse_arr=[]
    loss_reg_arr=[]
    val_chamfer = []
    val_frechet = []
    val_chamfer_bif = []
    val_frechet_bif = []
    val_chamfer_single = []
    val_frechet_single = []

    for epoch in range(config['EPOCHS']):
        start_time = time.perf_counter()

        loss_mask, loss_attraction, mse, reg = inner_train_loop( model,
                                 opt,
                                 data['train'], epoch )
        loss_mask_arr.append(loss_mask)
        loss_att_arr.append(loss_attraction)
        loss_mse_arr.append(mse)
        loss_reg_arr.append(reg)

        np.save(DIR + 'loss_mask', loss_mask_arr)
        np.save(DIR + 'loss_attraction', loss_att_arr)
        np.save(DIR + 'loss_mse', loss_mse_arr)
        np.save(DIR + 'loss_reg', loss_reg_arr)


        frechet, chamfer, frechet_bif, chamfer_bif, frechet_single, chamfer_single= inner_val_loop(model, data['val'], epoch)

        val_chamfer.append(chamfer)
        val_frechet.append(frechet)
        val_frechet_bif.append(frechet_bif)
        val_chamfer_bif.append(chamfer_bif)
        val_chamfer_single.append(chamfer_single)
        val_frechet_single.append(frechet_single)

        np.save(DIR + 'val_chamfer_mag_field', val_chamfer)
        np.save(DIR + 'val_frechet_mag_field', val_frechet)
        np.save(DIR + 'val_frechet_bif_mag_field', val_frechet_bif)
        np.save(DIR + 'val_chamfer_bif_mag_field', val_chamfer_bif)
        np.save(DIR + 'val_chamfer_single_mag_field', val_chamfer_single)
        np.save(DIR + 'val_frechet_single_mag_field', val_frechet_single)
    
        config['LOGGER'].info(f'epoch - {epoch+1} frechet dist - {frechet}')
        config['LOGGER'].info(f'epoch - {epoch+1} chamfer dist - {chamfer}')

        #clearml

        # config['TASK'].get_logger().report_scalar( title='epoch frechet score trace',
        #                                            series='val score',
        #                                            iteration=epoch,
        #                                            value=frechet)
        
        # config['TASK'].get_logger().report_scalar( title='epoch chamfer score trace',
        #                                            series='val score',
        #                                            iteration=epoch,
        #                                            value=chamfer )
        # config['TASK'].get_logger().report_scalar( title='length',
        #                                            series='val score',
        #                                            iteration=epoch,
        #                                            value=length )
        # config['TASK'].get_logger().report_scalar( title='epoch frechet bif score trace',
        #                                            series='val score',
        #                                            iteration=epoch,
        #                                            value=frechet_bif )
        # config['TASK'].get_logger().report_scalar( title='epoch chamfer single score trace',
        #                                            series='val score',
        #                                            iteration=epoch,
        #                                            value=chamfer_single )
        # config['TASK'].get_logger().report_scalar( title='epoch frechet single score trace',
        #                                            series='val score',
        #                                            iteration=epoch,
                                                #    value=frechet_single )

        if best_score > chamfer:
            best_score = chamfer
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_going_up = 0

            store(model)
        else:
            epochs_without_going_up += 1

        if epochs_without_going_up == config['STOP_EPOCHS']:
            break

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        config['LOGGER'].info(f'elapsed time {elapsed_time:.2f} s')
        config['LOGGER'].info(f'epoch without improve {epochs_without_going_up}')

    model.load_state_dict(best_state)

def load(model):
    state = torch.load(config['MODELNAMEINPUT'], map_location=config['DEVICE'])
    model.load_state_dict(state)

def store(model):
    state = model.state_dict()
    path = config['MODELNAME']

    torch.save(state, path)

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

@click.command()
# @click.option('--aug', '-aug', type=str, default='baseline')
@click.option('--datapath', '-dp', type=str, default = DIR + 'preprocessed_data')
@click.option('--modelname', '-mn', type=str, default='/home/kirill/library/model_name')
@click.option('--modelnameinput', '-mni', type=str, default='')
@click.option('--split_options_path', '-sop', type=str, default= DIR + 'split_options.yaml')
@click.option('--loader_options_path', '-lop', type=str, default=DIR + 'loader_options.yaml')
@click.option('--voxeling_options_path', '-vop', type=str, default=DIR + 'voxeling_options.yaml')
# @click.option('--debugname', '-dn', type=str, default='debug_name.json')
@click.option('--batch_size', '-bs', type=int, default=1)
@click.option('--n_iterations', '-ni', type=int, default=5000, help='The number of iteration per epoch')
@click.option('--epochs', '-e', type=int, default=10, help='The number of epoch per train loop')
# @click.option('--accumulation_step', '-as', type=int, default=1, help='The number of iteration to accumulate gradients')
@click.option('--stop_epochs', '-se', type=int, default=5)
@click.option('--learning_rate', '-lr', type=float, default=1e-3)
# @click.option('--backbone', '-bone', type=str, default='efficientnet-b0')
# @click.option('--inference_mode', '-im', type=click.Choice(['default', 'with_vertical_flip'], case_sensitive=False), default='default')
@click.option('--logger_type', '-lt', type=click.Choice(['stream', 'file'], case_sensitive=False), default='stream')
@click.option('--njobs', type=int, default=20, help='The number of jobs to run in parallel.')
@click.option('--checkpointing', '-chkp', is_flag=True, help='Whether checkpointing is used (https://pytorch.org/docs/stable/checkpoint.html)')
@click.option('--verbose', is_flag=True, default = True, help='Whether progress bars are showed')

def main(**kwargs):
    init_determenistic()

    init_global_config(**kwargs)
    config['N_CLASSES'] = 2

    for key in config:
        if key != 'LOGGER':
            config['LOGGER'].info(f'{key} {config[key]}')
            debugging_info[key.lower()] = str(config[key])

    data = load_data()

    config['LOGGER'].info(f'create model')
    model = my_unet()
    print(model)
    print(config)
    config['LOGGER'].info(f'fit model')
    fit(model, data)

    config['LOGGER'].info(f'store model')
    store(model)

if __name__ == '__main__':
    main()
