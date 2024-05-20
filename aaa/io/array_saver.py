import gzip
import numpy as np
import nibabel as nib


def save_masks(data_path, masks):
    masks = np.transpose(masks, (1, 2, 0))

    wmasks = nib.Nifti1Image(masks, np.eye(4))
    nib.save(wmasks, str(data_path))

def save_probs(data_path, probs):
    probs = np.transpose(probs, (1, 2, 0, -1))

    with gzip.GzipFile(data_path, 'w') as f:
        np.save(f, probs, allow_pickle=False)




