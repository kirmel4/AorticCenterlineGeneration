
def __refine(probs_masks_batch, refinement):
    if refinement == 'probabilities':
        output_batch = probs_masks_batch
    elif refinement == 'predictions':
        pred_masks_batch = probs_masks_batch.argmax(axis=1)
        output_batch = pred_masks_batch
    else:
        raise RuntimeError()

    return output_batch
