import cv2
import torch
import numpy as np

from scipy import special as sp

from aaa.inference.misc import __refine

def inference_default( model,
                       imgs_batch,
                       selections,
                       *,
                       refinement='predictions' ):
    logits_batch = model(imgs_batch)
    logits_batch = logits_batch.cpu().data.numpy()

    if refinement == 'logits':
        output_batch = logits_batch
    else:
        probs_masks_batch = sp.softmax(logits_batch, axis=1)
        output_batch = __refine(probs_masks_batch, refinement)

    return output_batch, selections

def inference_with_flip( model,
                         imgs_batch,
                         selections,
                         *,
                         axis=tuple(),
                         refinement='predictions' ):

    if len(axis) > 0:
        flipped_imgs_batch = torch.flip(imgs_batch, dims=axis)
    else:
        flipped_imgs_batch = imgs_batch

    flipped_probs_masks_batch, selections = inference_default( model,
                                                               flipped_imgs_batch,
                                                               selections,
                                                               refinement='probabilities' )

    if len(axis) > 0:
        flipped_probs_masks_batch = np.flip(flipped_probs_masks_batch, axis=axis)
    else:
        flipped_probs_masks_batch = flipped_probs_masks_batch

    probs_masks_batch = flipped_probs_masks_batch

    output_batch = __refine(probs_masks_batch, refinement)
    return output_batch, selections

def __inference_with_flips_generator(axises):
    def callee( model,
                imgs_batch,
                selections,
                *,
                refinement='predictions' ):

        probs_masks_batches = list()

        for axis in axises:
            flipped_probs_masks_batch, resampled_selections = inference_with_flip(
                model,
                imgs_batch,
                selections,
                axis=axis,
                refinement='probabilities'
            )

            probs_masks_batches.append(flipped_probs_masks_batch)

        probs_masks_batch = np.mean(probs_masks_batches, axis=0)
        selections = resampled_selections

        output_batch = __refine(probs_masks_batch, refinement)
        return output_batch, selections

    return callee

inference_with_vertical_flip = __inference_with_flips_generator([tuple(), (3, )])
