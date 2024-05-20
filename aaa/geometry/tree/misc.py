import cv2
import numpy as np

def __binary_assert(slice):
    assert slice.dtype == np.bool_

    uvalues = np.unique(slice)
    assert uvalues.size == 2
    assert 0 in uvalues and 1 in uvalues

# def get_square(slice, spacing):
#     __binary_assert(slice)

#     square = np.sum(slice) / spacing**2

#     return { 'square': square,
#              'square unit': 'mm^2' }

# def get_ellipse_diameter(slice, spacing):
#     __binary_assert(slice)

#     contours, _ = cv2.findContours( slice.astype(np.uint8),
#                                     cv2.RETR_EXTERNAL,
#                                     cv2.CHAIN_APPROX_NONE )

#     assert len(contours) == 1

#     center, diameters, orientation = cv2.fitEllipse(contours[0])

#     return { '__center': center,
#              '__diameters': diameters,
#              '__orientation': orientation,
#              'ellipse diameter': diameters[1] * spacing,
#              'ellipse diameter unit': 'mm' }

# def get_maximal_diameter(slice, spacing, epsilon=np.pi/180):
#     __binary_assert(slice)

#     contours, _ = cv2.findContours( slice.astype(np.uint8),
#                                     cv2.RETR_EXTERNAL,
#                                     cv2.CHAIN_APPROX_NONE )

#     assert len(contours) == 1

#     contour = contours[0]

#     moments = cv2.moments(contour)
#     cpoint = np.array([moments['m10'], moments['m01']]) / moments['m00']

#     contour = np.array(contour)[:, 0]

#     maximal_diameter = 0
#     maximal_xpoint = None
#     maximal_ypoint = None

#     for xpoint in contour:
#         for ypoint in contour:
#             vdiameter = ypoint - xpoint
#             vhalfdiameter = cpoint - xpoint

#             cosine = np.sum(vdiameter * vhalfdiameter) / ( np.linalg.norm(vdiameter) * np.linalg.norm(vhalfdiameter) )
#             angle = np.arccos(cosine)

#             if angle < epsilon:
#                 if maximal_diameter < np.linalg.norm(vdiameter):
#                     maximal_diameter = np.linalg.norm(vdiameter)
#                     maximal_xpoint = xpoint
#                     maximal_ypoint = ypoint

#     return { '__xpoint': maximal_xpoint,
#              '__ypoint': maximal_ypoint,
#              'maximal diameter': maximal_diameter * spacing,
#              'maximal diameter unit': 'mm' }

# def get_hydraulic_diameter(slice, spacing):
#     __binary_assert(slice)

#     square = get_square(slice, 1.)['square']

#     perimeter = get_perimeter(slice.astype(int))
#     diameter = 4 * square / perimeter

#     return { 'hydraulic diameter': diameter * spacing,
#              'hydraulic diameter unit': 'mm' }

# def get_slice_density(slice, imgslice):
#     __binary_assert(slice)

#     mass = np.sum(imgslice[slice])
#     volume = np.sum(slice)

#     return { 'slice density': mass / volume,
#              'slice density unit': 'HU per mm^3' }

# def get_wall_thickness(slice, point, spacing, *, angular_step=AngularStep.ANGLE_HALF_DEGREE):
#     """ NOTE https://stackoverflow.com/questions/64077984/finding-the-max-and-min-distance-between-two-polygons-using-opencv

#         Calculate wall thickness

#         Assumed slice is correct segmented
#     """
#     __binary_assert(slice)

#     distmap = cv2.distanceTransform( slice.astype(np.uint8),
#                                      distanceType=cv2.DIST_L2,
#                                      maskSize=cv2.DIST_MASK_5 )

#     if angular_step == AngularStep.ANGLE_HALF_DEGREE:
#         size = (512, 720)
#         degree_multiplicator = 0.5
#     elif angular_step == AngularStep.ANGLE_ONE_DEGREE:
#         size = (512, 360)
#         degree_multiplicator = 1.
#     elif angular_step == AngularStep.ANGLE_TWO_DEGREE:
#         size = (512, 180)
#         degree_multiplicator = 2.
#     else:
#         raise RuntimeError(f'Unknown angular_step value - {angular_step}')

#     point = tuple(np.array(point)[[1, 0]])

#     distmap_in_polar = cv2.warpPolar( distmap,
#                                       dsize=size,
#                                       center=point,
#                                       maxRadius=256//2,
#                                       flags=cv2.INTER_CUBIC+cv2.WARP_POLAR_LINEAR )

#     distances_in_polar = np.amax(distmap_in_polar, axis=1)

#     max_thickness = 2 * np.amax(distances_in_polar) * spacing
#     angle_max_thickness = degree_multiplicator*np.argmax(distances_in_polar)

#     min_thickness = 2 * np.amin(distances_in_polar) * spacing
#     angle_min_thickness = degree_multiplicator*np.argmin(distances_in_polar)

#     return {
#         'maximal thickness': max_thickness,
#         '__angle_max_thickness': angle_max_thickness,
#         'minimal thickness': min_thickness,
#         '__angle_min_thickness': angle_min_thickness,
#         'thickness unit': 'mm'
#     }
