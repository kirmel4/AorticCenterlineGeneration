def _average_value_checker(args):
    if args['average'] not in {'binary', 'macro'}:
        raise ValueError('average value is incorrect')

def _reduction_value_checker(args):
    if args['reduction'] not in {'sum', 'mean'}:
        raise ValueError('reduction value is incorrect')

def _3d_shape_asserter(args):
    if not (len(args['y_masks_batch'].shape) == 4):
        raise RuntimeError(f"y_masks_batch length is {len(args['y_masks_batch'].shape)} not 4")

    if not (len(args['pred_logits_batch'].shape) == 5):
        raise RuntimeError(f"pred_logits_batch length is {len(args['pred_logits_batch'].shape)} not 5")

    if not (args['y_masks_batch'].shape[0] == args['pred_logits_batch'].shape[0]):
        raise RuntimeError(f"y_masks_batch batch size is {args['y_masks_batch'].shape[0]}, "
                           f"but pred_logits_batch batch size is {args['pred_logits_batch'].shape[0]}")

    assert args['y_masks_batch'].shape[1] == args['pred_logits_batch'].shape[2]
    assert args['y_masks_batch'].shape[2] == args['pred_logits_batch'].shape[3]
    assert args['y_masks_batch'].shape[3] == args['pred_logits_batch'].shape[4]
