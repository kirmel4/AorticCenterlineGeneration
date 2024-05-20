import pytest

import torch
import numpy as np

from aaa.losses.threedim import cross_entropy_with_logits_loss

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.cross_entropy_based_loss
def test_cross_entropy_loss_CASE_reduction_incorrect_value():
    y_masks_batch = np.zeros((1, 1, 1, 1))
    pred_logits_batch = np.zeros((1, 1, 1, 1, 1))

    with pytest.raises(ValueError, match='reduction value is incorrect') as e:   
        cross_entropy_with_logits_loss(y_masks_batch, pred_logits_batch, reduction='incorrect_value')

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.cross_entropy_based_loss
def test_cross_entropy_loss_CASE_y_masks_batch_wrong_shape():
    y_masks_batch = np.zeros((1, 1, 1))
    pred_logits_batch = np.zeros((1, 1, 1, 1, 1))

    with pytest.raises(RuntimeError, match=r'y_masks_batch length is \d+ not 4$') as e:   
        cross_entropy_with_logits_loss(y_masks_batch, pred_logits_batch)

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.cross_entropy_based_loss
def test_cross_entropy_loss_CASE_pred_logits_batch_wrong_shape():
    y_masks_batch = np.zeros((1, 1, 1, 1))
    pred_logits_batch = np.zeros((1, 1, 1, 1))

    with pytest.raises(RuntimeError, match=r'pred_logits_batch length is \d+ not 5$') as e:   
        cross_entropy_with_logits_loss(y_masks_batch, pred_logits_batch)

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.cross_entropy_based_loss
def test_cross_entropy_loss_CASE_reduction_incorrect_value():
    y_masks_batch = np.zeros((2, 1, 1, 1))
    pred_logits_batch = np.zeros((1, 1, 1, 1, 1))

    with pytest.raises(RuntimeError, match=r'y_masks_batch batch size is \d+, but pred_logits_batch batch size is \d+$') as e:   
        cross_entropy_with_logits_loss(y_masks_batch, pred_logits_batch)

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.cross_entropy_based_loss
def test_cross_entropy_loss_CASE_two_classes():
    y_masks_batch = np.array([ [ [ [0, 1, 1], [1, 0, 1] ] ] ])

    pred_logits_batch = np.array([
      [ [ [ [0.9, 0.2, 0.3], [0.4, 0.6, 0.7] ] ] ,
        [ [ [0.1, 0.8, 0.7], [0.6, 0.4, 0.3] ] ] ]
    ])

    softmax = lambda x, y: np.exp(x) / np.exp(y).sum()

    y_masks_batch = torch.tensor(y_masks_batch)
    pred_logits_batch = torch.tensor(pred_logits_batch)

    loss = cross_entropy_with_logits_loss(
        y_masks_batch,
        pred_logits_batch,
        average='binary',
        activation='softmax',
        reduction='sum'
    )

    expected = - ( np.log( softmax(0.9, [0.9, 0.1]) ) +
                   np.log( softmax(0.8, [0.2, 0.8]) ) +
                   np.log( softmax(0.7, [0.3, 0.7]) ) +
                   np.log( softmax(0.6, [0.4, 0.6]) ) +
                   np.log( softmax(0.6, [0.6, 0.4]) ) +
                   np.log( softmax(0.3, [0.3, 0.7]) ) ) / 6

    assert pytest.approx(loss.item()) == expected

@pytest.mark.loss
@pytest.mark.threedim_loss
@pytest.mark.cross_entropy_based_loss
def test_cross_entropy_loss_CASE_one_class():
    y_masks_batch = np.array([ [ [ [0., 1.], [1., 0.] ] ] ])
    pred_logits_batch = np.array([
       [ [ [ [0.1, 0.9], [0.6, 0.4] ] ] ]
    ])

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    y_masks_batch = torch.tensor(y_masks_batch)
    pred_logits_batch = torch.tensor(pred_logits_batch)

    loss = cross_entropy_with_logits_loss(
        y_masks_batch,
        pred_logits_batch,
        average='binary',
        activation='sigmoid',
        reduction='sum'
    )
    expected = - ( np.log( 1 - sigmoid(0.1) ) +
                   np.log( sigmoid(0.9) ) +
                   np.log( sigmoid(0.6) ) +
                   np.log( 1 - sigmoid(0.4) ) ) / 4

    assert pytest.approx(loss.item()) == expected
