import torch
import numpy as np
import nibabel as nib

from collections import OrderedDict
from torch.utils.data import Dataset

from aaa.utils import torch_float, torch_long, io

class aaaIGzDataset(Dataset):
    """Dataset for loading CT images from nii.gz format
    """
    def __init__(self, datapath, names, *, channels=None):
        """
            :NOTE:
                imgs must be called imgs.nii.gz

                format of channels:

                { # yaml notation
                    CHANNEL_NAME_1:
                        MIN_HU: VALUE
                        MAX_HU: VALUE
                    CHANNEL_NAME_2:
                        ...
                }

            :args:
                datapath (pathlib.Path): directiry path with data
                names (list of str): names of data to load
                channels (dict, see NOTE): option to online form channels for imgs
        """

        self.channels = channels

        self.imgs = OrderedDict()

        self.keys = list()
        self.shapes = list()

        for name in names:
            image = nib.load(datapath + '/' + name + '/' + 'imgs.nii.gz')
            idata = image.dataobj[:]

            self.imgs[name] = idata

            self.keys.append(name)
            self.shapes.append(idata.shape)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        key, selector = idx

        voxel = self.imgs[key][selector]
        voxel = io.split_images(voxel[:, :, None], self.channels) # :
        voxel = np.moveaxis(voxel, -1, 0)

        return voxel, key, selector

    @staticmethod
    def collate_fn(batch):
        voxels, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = torch_float(voxels, torch.device('cpu'))

        return voxels, selections

class aaaIMGzDataset(aaaIGzDataset):
    """Dataset for loading CT images and segmentation masks from nii.gz format
    """
    def __init__(self, datapath, names, *, channels=None):
        super().__init__(datapath, names, channels=channels)

        self.masks = dict()

        for name in names:
            mask = nib.load(datapath + '/' + name + '/' + 'masks.nii.gz')
            mdata = mask.dataobj[:]

            self.masks[name] = mdata

    def __getitem__(self, idx):
        voxel, key, selector = aaaIGzDataset.__getitem__(self, idx)
        mask = self.masks[key][selector]

        return voxel, mask, key, selector

    @staticmethod
    def collate_fn(batch):
        voxels, masks, keys, selectors = zip(*batch)
        selections = [*zip(keys, selectors)]

        voxels = torch_float(voxels, torch.device('cpu'))
        masks = torch_long(masks, torch.device('cpu'))

        return voxels, masks, selections

class aaaIMAGzDataset(aaaIMGzDataset):
    ...
