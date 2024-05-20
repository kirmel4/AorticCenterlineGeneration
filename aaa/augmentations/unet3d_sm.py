import cv2
import albumentations as A

train_aug = A.Compose([ A.VerticalFlip(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.OneOf([ A.GaussNoise( mean=0,
                                                var_limit=(0.01, 0.01),
                                                p=0.8 ),
                                  A.GaussianBlur( blur_limit=(3, 11),
                                                  sigma_limit=(0.3, 0.3),
                                                  p=0.8
                                                )
                        ], p=0.9),
                        A.ShiftScaleRotate( shift_limit_x=(-0.1, 0.1),
                                            shift_limit_y=(-0.1, 0.1),
                                            scale_limit=(-0.2, 0.2),
                                            rotate_limit=(-15, 15),
                                            interpolation=cv2.INTER_LINEAR,
                                            border_mode=cv2.BORDER_REPLICATE,
                                            p=0.8 ),
                        A.OneOf([ A.ElasticTransform( alpha=1,
                                                      sigma=50,
                                                      alpha_affine=10,
                                                      interpolation=cv2.INTER_LINEAR,
                                                      border_mode=cv2.BORDER_REPLICATE,
                                                      value=0,
                                                      mask_value=0,
                                                      approximate=False,
                                                      p=0.8 ),
                                  A.GridDistortion( num_steps=4,
                                                    distort_limit=0.3,
                                                    interpolation=cv2.INTER_LINEAR,
                                                    border_mode=cv2.BORDER_REPLICATE,
                                                    p=0.8 )
                        ], p=0.9),
])

infer_aug = A.Compose([ ])

pre_aug = A.Compose([ A.CenterCrop( height=256,
                                    width=256,
                                    p=1.0 )
                    ])

inverse_infer_aug = (lambda value, height, width:
    A.Compose([ A.PadIfNeeded( min_height=height,
                               min_width=width,
                               border_mode=cv2.BORDER_CONSTANT,
                               value=value,
                               p=1.0 ) ]))
