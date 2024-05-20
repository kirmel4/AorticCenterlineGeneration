import cv2
import torch
import numpy as np

from scipy import special as sp

from aaa.inference.misc import __refine
from aaa.utils import inference as misc

def __augment(logits_batch, aug):
    auged_logits_batch = list()

    for logits in logits_batch:
        auged_logits = list()

        for channel in logits:
            auged = aug(image=channel)
            aug_channel = auged['image']

            auged_logits.append(aug_channel)
        auged_logits = np.array(auged_logits)
        auged_logits_batch.append(auged_logits)

    return np.array(auged_logits_batch)

def inference_default( model,
                       imgs_batch,
                       *,
                       refinement='predictions',
                       aug=dict,
                       inverse_resample_factors=False,
                       resample=False,
                       resample_factors=(1., 1.),
                       resample_interpolation=cv2.INTER_LINEAR ):
    logits_batch = model(imgs_batch)
    logits_batch = logits_batch.cpu().data.numpy()

    logits_batch = __augment(logits_batch, aug)

    if resample:
        if inverse_resample_factors:
            resample_factors = tuple(1. / factor for factor in resample_factors)

        resampled_logits_batch = list()

        for channel_idx in np.arange(logits_batch.shape[1]):
            resampled_channel = misc.resample_imgs( logits_batch[:, channel_idx],
                                                    resample_factors )

            resampled_logits_batch.append(resampled_channel)

        resampled_logits_batch = np.stack(resampled_logits_batch, axis=1)
        logits_batch = resampled_logits_batch

    if refinement == 'logits':
        output_batch = logits_batch
    else:
        probs_masks_batch = sp.softmax(logits_batch, axis=1)
        output_batch = __refine(probs_masks_batch, refinement)

    return output_batch

def inference_with_vertical_flip( model,
                                  imgs_batch,
                                  *,
                                  refinement='predictions',
                                  aug=dict,
                                  inverse_resample_factors=False,
                                  resample=False,
                                  resample_factors=(1., 1.),
                                  resample_interpolation=cv2.INTER_LINEAR ):
    kwargs = {
        'aug': aug,
        'inverse_resample_factors': inverse_resample_factors,
        'resample': resample,
        'resample_factors': resample_factors,
        'resample_interpolation': resample_interpolation
    }

    default_probs_masks_batch = inference_default( model,
                                                   imgs_batch, 
                                                   refinement='probabilities',
                                                   **kwargs )

    vertical_flipped_imgs_batch = torch.flip(imgs_batch, dims=(2, ))
    vertical_flipped_probs_masks_batch = inference_default( model,
                                                            vertical_flipped_imgs_batch, 
                                                            refinement='probabilities',
                                                            **kwargs )

    vertical_flipped_probs_masks_batch = np.flip(vertical_flipped_probs_masks_batch, axis=(2, ))

    probs_masks_batch = 0.5 * (default_probs_masks_batch + vertical_flipped_probs_masks_batch)

    output_batch = __refine(probs_masks_batch, refinement)
    return output_batch
