import numpy as np

from itertools import product, chain
from joblib import Parallel, delayed
from torch.utils.data import Sampler

def voxel_random_selector(voxel_shape, case_keys, shapes, n_count=1):
    assert type(case_keys) in {list, tuple}
    assert len(case_keys) == len(shapes)

    n_cases = np.arange(len(case_keys))

    for _ in np.arange(n_count):
        case_idx = np.random.choice(n_cases)

        case_key = case_keys[case_idx]
        case_shape = shapes[case_idx]

        selector = ()

        for voxel_size, case_size in zip(voxel_shape, case_shape):
            if voxel_size is None:
                selector = (*selector, slice(0, None))
            else:
                assert voxel_size <= case_size

                if voxel_size == case_size:
                    coord = 0
                else:
                    coord = np.random.choice(case_size - (voxel_size - 1))

                selector = (*selector, slice(coord, coord + voxel_size))

        yield case_key, selector

class VoxelRandomSampler(Sampler):
    def __init__(self, voxel_shape, case_keys, shapes, n_count=1):
        self.voxel_shape = voxel_shape
        self.case_keys = case_keys
        self.shapes = shapes
        self.n_count = n_count

    def __iter__(self):
        yield from voxel_random_selector( self.voxel_shape,
                                          self.case_keys,
                                          self.shapes,
                                          self.n_count )

    def __len__(self):
        return self.n_count

def voxel_sequential_selector(voxel_shape, case_keys, shapes, steps):
    assert type(case_keys) in {list, tuple}
    assert len(voxel_shape) == len(steps)

    for idx, case_key in enumerate(case_keys):
        ranges = ()

        for step, voxel_size, case_size in zip(steps, voxel_shape, shapes[idx]):
            assert case_size >= voxel_size

            range_ = chain(np.arange(0, case_size - voxel_size, step), (case_size - voxel_size ,))
            ranges = (*ranges, range_)

        for point in product(*ranges):
            selector = ()

            for coord, voxel_size in zip(point, voxel_shape):
                selector = (*selector, slice(coord, coord+voxel_size))

            yield case_key, selector

class VoxelSequentialSampler(Sampler):
    def __init__(self, voxel_shape, case_keys, shapes, steps):
        self.voxel_shape = voxel_shape
        self.case_keys = case_keys
        self.shapes = shapes
        self.steps = steps

    def __iter__(self):
        yield from voxel_sequential_selector( self.voxel_shape,
                                              self.case_keys,
                                              self.shapes,
                                              self.steps )

    def __len__(self):
        return len([*voxel_sequential_selector( self.voxel_shape,
                                                self.case_keys,
                                                self.shapes,
                                                self.steps )])

def __voxel_batch_generator_without_parallelization(imgs, masks, selections):
    voxels_batch = list()
    masks_batch = list()

    for case_key, selector in selections:
        voxels_batch.append(imgs[case_key][selector])
        masks_batch.append(masks[case_key][selector])

    return np.array(voxels_batch), np.array(masks_batch)

def __select(array, case_key, selector):
    return array[case_key][selector]

def __voxel_batch_generator_with_parallelization(imgs, masks, selections, *, njobs):
    voxels_batch = Parallel(n_jobs=njobs)(delayed(__select)(imgs, case_key, selector) for case_key, selector in selections)
    masks_batch = Parallel(n_jobs=njobs)(delayed(__select)(masks, case_key, selector) for case_key, selector in selections)

    return np.array(voxels_batch), np.array(masks_batch)

def voxel_batch_generator(imgs, masks, selections, *, njobs=1):
    if njobs == 1:
        return __voxel_batch_generator_without_parallelization(imgs, masks, selections)
    else:
        return __voxel_batch_generator_with_parallelization(imgs, masks, selections, njobs=njobs)

    return np.array(voxels_batch), np.array(masks_batch)
