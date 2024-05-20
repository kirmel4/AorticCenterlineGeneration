import types
import albumentations as A

from functools import partial
from multiprocessing import Pool

def __apply_with_params_loop(args, self, params):
    key, arg = args

    if arg is not None:
        target_function = self._get_target_function(key)
        target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
        return (key, target_function(arg, **dict(params, **target_dependencies)))
    else:
        return (key, None)

def __apply_with_params_pool(njobs):
    pool = Pool(njobs)

    def apply_with_params(self, params, force_apply = False, **kwargs):
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)

        loop = partial(__apply_with_params_loop, self=self, params=params)
        res = pool.map(loop, kwargs.items())

        return dict(res)

    return apply_with_params

def aug_parallelization(aug, njobs=1):
    if isinstance(aug, A.core.composition.BaseCompose):
        for item in aug:
            aug_parallelization(item, njobs=njobs)
    elif isinstance(aug, A.core.transforms_interface.BasicTransform):
        aug.apply_with_params = types.MethodType(__apply_with_params_pool(njobs), aug)
    else:
        raise RuntimeError()

def __targets_setter(self, arg):
    self._targets = arg

def aug_uda(aug):
    if isinstance(aug, A.core.composition.BaseCompose):
        for item in aug: aug_uda(item)
    elif isinstance(aug, A.core.transforms_interface.DualTransform):
        aug.__class__._targets = {
            'image': aug.__class__.apply,
            'mask': aug.__class__.apply_to_mask,
            'masks': aug.__class__.apply_to_masks,
            'bboxes': aug.__class__.apply_to_bboxes,
            'keypoints': aug.__class__.apply_to_keypoints
        }

        aug.__class__.targets = property(fget=lambda self: self._targets, fset=__targets_setter)

        aug.targets = {
            'image': aug.__class__.apply,
            'mask': aug.__class__.apply,
            'masks': aug.__class__.apply_to_masks,
            'bboxes': aug.__class__.apply_to_bboxes,
            'keypoints': aug.__class__.apply_to_keypoints
        }
    elif isinstance(aug, A.core.transforms_interface.BasicTransform):
        pass
    else:
        raise RuntimeError()
