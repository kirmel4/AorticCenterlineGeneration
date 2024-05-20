import pytest

import torch
import numpy as np

from aaa.losses.threedim import dice_with_logits_loss

INAPPROPRIATE_PARAMETERS = [
    (torch.zeros((1, 1, 1, 1, 1)), 'binary', 'softmax', 'mean', dict()),
    (torch.zeros((1, 2, 1, 1, 1)), 'binary', 'sigmoid', 'mean', dict()),
    (torch.zeros((1, 1, 1, 1, 1)), 'binary', 'none', 'mean', dict()),
    (torch.zeros((1, 1, 1, 1, 1)), 'macro', 'none', 'mean', dict()),
    (torch.zeros((1, 1, 1, 1, 1)), 'macro', 'sigmoid', 'mean', dict()),
    (torch.zeros((1, 1, 1, 1, 1)), 'binary', 'sigmoid', 'mean', dict(unexpected=None)),
]

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.confusion_based_loss
@pytest.mark.parametrize('pred_logits_batch, average, activation, reduction, parameters', INAPPROPRIATE_PARAMETERS)
def test_dice_with_logits_loss_CASE_inappropriate_parameters(pred_logits_batch, average, activation, reduction, parameters):
    y_masks_batch = torch.zeros((1, 1, 1, 1))

    with pytest.raises( (RuntimeError, ValueError)) as e:
        dice_with_logits_loss( y_masks_batch,
                               pred_logits_batch,
                               average=average,
                               activation=activation,
                               reduction=reduction,
                               parameters=parameters )

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.confusion_based_loss
def test_dice_with_logits_loss_CASE_binary_AND_sigmoid():
    SMOOTH = 1

    y_masks_batch = np.array([ [ [ [0, 1], [1, 0] ] ] ])
    pred_logits_batch = np.array([ [ [ [ [0.9, 0.2], [0.4, 0.6] ] ] ] ])

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    y_masks_batch = torch.tensor(y_masks_batch)
    pred_logits_batch = torch.tensor(pred_logits_batch)

    loss = dice_with_logits_loss(
        y_masks_batch,
        pred_logits_batch,
        average='binary',
        activation='sigmoid',
        reduction='sum',
        hyperparameters=dict(smooth=SMOOTH)
    )

    tp = 0 * sigmoid(0.9) + 1 * sigmoid(0.2) + 1 * sigmoid(0.4) + 0 * sigmoid(0.6)
    fp = (1 - 0) * sigmoid(0.9) + (1 - 1) * sigmoid(0.2) + (1 - 1) * sigmoid(0.4) + (1 - 0) * sigmoid(0.6)
    fn = 0 * (1 - sigmoid(0.9)) + 1 * (1 - sigmoid(0.2)) + 1 * (1 - sigmoid(0.4)) + 0 * (1 - sigmoid(0.6))

    expected = 1 - (2 * tp + SMOOTH) / ( tp + fp + tp + fn + SMOOTH)

    assert pytest.approx(loss.item()) == expected

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.confusion_based_loss
def test_dice_with_logits_loss_CASE_binary_AND_softmax():
    SMOOTH = 1

    y_masks_batch = np.array([ [ [ [0, 1], [1, 0] ] ] ])
    pred_logits_batch = np.array([
      [ [ [ [0.1, 0.8], [0.6, 0.4] ] ],
        [ [ [0.9, 0.2], [0.4, 0.6] ] ] ]
    ])

    softmax = lambda x, y: np.exp(x) / np.exp(y).sum()

    y_masks_batch = torch.tensor(y_masks_batch)
    pred_logits_batch = torch.tensor(pred_logits_batch)

    loss = dice_with_logits_loss(
        y_masks_batch,
        pred_logits_batch,
        average='binary',
        activation='softmax',
        reduction='sum',
        hyperparameters=dict(smooth=SMOOTH)
    )

    tp = 0 * softmax(0.9, [0.9, 0.1]) + \
         1 * softmax(0.2, [0.2, 0.8]) + \
         1 * softmax(0.4, [0.4, 0.6]) + \
         0 * softmax(0.6, [0.6, 0.4])

    fp = (1 - 0) * softmax(0.9, [0.9, 0.1]) + \
         (1 - 1) * softmax(0.2, [0.2, 0.8]) + \
         (1 - 1) * softmax(0.4, [0.4, 0.6]) + \
         (1 - 0) * softmax(0.6, [0.6, 0.4])

    fn = 0 * (1 - softmax(0.9, [0.9, 0.1])) + \
         1 * (1 - softmax(0.2, [0.2, 0.8])) + \
         1 * (1 - softmax(0.4, [0.4, 0.6])) + \
         0 * (1 - softmax(0.6, [0.6, 0.4]))

    expected = 1 - (2 * tp + SMOOTH) / ( tp + fp + tp + fn + SMOOTH)

    assert pytest.approx(loss.item()) == expected
