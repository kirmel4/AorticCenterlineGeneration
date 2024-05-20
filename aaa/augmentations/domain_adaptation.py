import numpy as np

from skimage.transform import radon, iradon
from albumentations.core.transforms_interface import ImageOnlyTransform

class FiltredBackProjectionNormalization(ImageOnlyTransform):
    def __init__(
        self,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(always_apply, p)

    def apply(self, image, **params):
        nrows, ncols = image.shape
        assert nrows == ncols

        radius = nrows // 2
        xpr, ypr = np.mgrid[:nrows, :ncols] - radius
        outside_mask = (xpr ** 2 + ypr ** 2) > radius ** 2

        outside_I = image[0, 0]
        image[outside_mask] = 0

        theta = np.linspace(0, 180, nrows, endpoint=False)
        sinogram = radon(image, theta)

        reconstructed_image = iradon(sinogram, filter_name='ramp')
        reconstructed_image[outside_mask] = outside_I

        return np.ascontiguousarray(np.transpose(reconstructed_image))

    def get_transform_init_args_names(self):
        return tuple()
