from aaa.geometry.descriptors.descriptor import Descriptor
from aaa.geometry.descriptors.descriptor3d import Descriptor3D
import nibabel as nib
import numpy as np
import nibabel as nib
import glob
import torch
import matplotlib.pyplot as plt
from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler
from aaa.geometry.enum import Classes
import random
import os
from scipy.spatial import KDTree
from tqdm import tqdm
import gc

DIR = './'

# Data exapmple:
#data
#└── data_1
#      ├── masks_LK.nii.gz
#      └── imgs.nii.gz'
#└── data_2
#...
#data in x-y-z format, script converts it to z-x-y
os.mkdir(DIR +'preprocessed_data')

def get_field(probs, points):
    kdtree = KDTree(points)
    field = np.zeros((probs.shape[0], probs.shape[1], probs.shape[2], 3))
    for z in tqdm(range(probs.shape[0])):
        for x in range(probs.shape[1]):
            for y in range(probs.shape[2]):
                minimal_idx = kdtree.query([z, x , y], 1)[1]
                #field[z][x][y] = [points[minimal_idx][0] - z, points[minimal_idx][1] - x ,points[minimal_idx][2] - y]
                field[z][x][y] = np.add(points[minimal_idx], np.array([-z, -x, -y])).round().astype(int)
    return field
for dir in tqdm(os.listdir(DIR +'data')):
    try:
        print(dir)
        image = nib.load(DIR + 'data/'+dir + '/imgs.nii.gz').get_fdata()
        mask = nib.load(DIR + 'data/'+ dir + '/masks_LK.nii.gz').get_fdata()
        masks_z_not_null = np.sum(mask, axis=(0, 1))
        z_ids, = np.where(masks_z_not_null > 0)
        z_min = np.min(z_ids)
        z_max = np.max(z_ids)
        image = image[128:512-128, 128:512-128 ,z_min:z_max]
        mask= mask[128:512-128, 128:512-128, z_min:z_max]
        if not os.path.exists(DIR +'preprocessed_data/' + dir):
            os.mkdir(DIR +'preprocessed_data/' + dir)
        if not os.path.exists(DIR +'preprocessed_data/' + dir +f'/{dir}_img.npz'):
            np.savez_compressed(DIR +'preprocessed_data/' + dir +f'/{dir}_img', data = np.moveaxis(image, -1, 0))
        zeros = np.where((mask  == 1), 4, mask)
        zeros = np.where((zeros == 0) | (zeros == 2) | (zeros == 3), 1, zeros)
        zeros = np.where((zeros== 4), 0, zeros)
        ones = np.where((mask == 2) | (mask == 3), 0, mask)
        segmentation_mask = np.array([zeros,ones])
        if not os.path.exists(DIR +'preprocessed_data/' + dir +f'/{dir}_segmentation.npz'):
            np.savez_compressed(DIR +'preprocessed_data/' + dir +f'/{dir}_segmentation', data = np.swapaxes(segmentation_mask, -1, 0))
        if not os.path.exists(DIR +'preprocessed_data/' + dir +f'/{dir}_attraction.npz'):
            twos = np.where((zeros== 1) , 0, zeros)
            threes = twos.copy()
            imgs = np.array(image)
            probs = np.array([zeros, ones, twos, threes])
            probs = np.moveaxis(probs, 0,3)
            imgs = np.moveaxis(imgs, 2, 0)
            probs = np.moveaxis(probs, 2,0)
            descriptor = Descriptor3D(imgs, probs)
            points = np.array(descriptor._ps)
            #print('Mask shape: ' + str(probs.shape))
            #print('Centerline points amount: ' + str(points.shape))
            attraction_field = get_field(probs, points)
            np.savez_compressed(DIR +'preprocessed_data/' + dir +f'/{dir}_attraction', data = attraction_field)
    except Exception as e:
        print(str(e))
        continue