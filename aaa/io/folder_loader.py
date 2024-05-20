import cv2
import gzip
import json
import warnings
import numpy as np

from medpy import io

from aaa.utils import io as misc

def load(data_dir, *, verbose=False,
                      load_without_masks=False,
                      load_probs=False,
                      load_metainfo=False,
                      drop_empty_slices=True,
                      channels=None ):
    """ NOTE dicom files have pamametrs as RescaleIntercept, RescaleSlope
        used for scaling raw intensities to Hounsfield Unit (HU)

        medpy AUTOMATICALLY convert raw intensities to HU

        but medpy does not treat outside pixels
        usually they converted to Air HU (-1000 absolute value)

        Expected dir tree: 

        |-- <data_dir>
            |-- images
                |-- <NAME1>
                    |-- <CT-images in DICOM format>
                |-- <NAME2>
                    |-- <CT-images in DICOM format>
                ...
            |-- masks
                |-- <NAME1>
                    |-- masks.vtk
                |-- <NAME2>
                    |-- masks.vtk
                ...

        Output orientation of images is zxy(channel)

            z
            ^
            |
            |
            .----> y
           /
          /
         v
        x

        z - from foot to head
        y - from chest to back
    """
    if verbose == True:
        from tqdm import tqdm
        verboser = tqdm
    else:
        verboser = lambda x: x

    assert data_dir.is_dir()

    images_dir = data_dir / 'images'
    assert images_dir.is_dir()

    if not load_without_masks:
        masks_dir = data_dir / 'masks'
        assert masks_dir.is_dir()

    if load_metainfo:
        metainfo_dir = data_dir / 'metainfo'
        assert metainfo_dir.is_dir()

    data = {}

    for path in verboser(images_dir.glob('*')):
        if path.is_dir():
            subdata = { }

            name = path.stem
            images, header = io.load(str(path))

            images = misc.normalize_outside_pixels(images)

            images = np.transpose(images, (2 ,0 ,1))
            images = images.astype(np.float32)

            if not load_without_masks:
                masks_path = masks_dir / name / 'masks.vtk'
                assert (masks_path).is_file()

                masks, _ = io.load(str(masks_path))
                masks = np.transpose(masks, (2 ,0 ,1))

                if load_probs:
                    probs_path = masks_dir / name / 'probs.npy.gz'
                    assert (probs_path).is_file()

                    with gzip.GzipFile(probs_path, 'r') as f:
                        probs = np.load(f)

                    probs = np.transpose(probs, (2, 0, 1, -1))

                if drop_empty_slices and not load_metainfo:
                    if load_probs:
                        images, masks, probs = misc.drop_empty_slices_with_probs(images, masks, probs)
                    else:
                        images, masks = misc.drop_empty_slices(images, masks)

                assert images.shape == masks.shape

            images = np.expand_dims(images, axis=-1)

            if channels:
                images = misc.split_images(images, channels)

            spacing = header.get_info_consistent(3)[0]
            subdata['metainfo'] = {
                'original xyz spacing': (spacing[1], spacing[0], spacing[2])
            }

            if load_metainfo:
                metainfo_path = metainfo_dir / name / 'metainfo.json'
                assert (metainfo_path).is_file()

                with open(metainfo_path, 'r') as f:
                    metainfo = json.load(f)

                subdata['metainfo'].update(metainfo)

                if not load_without_masks:
                    if load_probs:
                        images, masks, probs = misc.drop_empty_slices_with_probs(images, masks, probs, range=metainfo['diapason'])
                    else:
                        images, masks = misc.drop_empty_slices(images, masks, range=metainfo['diapason'])

            subdata['imgs'] = images
            _, height, width, _ = images.shape

            subdata['metainfo']['height'] = height
            subdata['metainfo']['width'] = width

            if not load_without_masks:
                subdata['masks'] = masks

                if load_probs:
                    subdata['probs'] = probs

            data[name] = subdata
        else:
            warnings.warn(f'path - {path} is not directory', RuntimeWarning, stacklevel=2)

    return data
