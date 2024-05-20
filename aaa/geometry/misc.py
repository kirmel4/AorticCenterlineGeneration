import cv2
import numpy as np

def __respace_iterate_zero_axis(data, fx, fy, interpolation):
    rdata = list()

    for idx in np.arange(data.shape[0]):
        rdatum = cv2.resize( data[idx],
                             None,
                             fx=fx,
                             fy=fy,
                             interpolation=interpolation )

        rdata.append(rdatum)

    return np.array(rdata)

def __respace_iterate_second_axis(data, fx, fy, interpolation):
    rdata = list()

    for idx in np.arange(data.shape[2]):
        rdatum = cv2.resize( data[:,:,idx],
                             None,
                             fx=fx,
                             fy=fy,
                             interpolation=interpolation )

        rdata.append(rdatum)

    rdata = np.array(rdata)
    rdata = rdata.transpose(1, 2, 0)

    return rdata

def respace(data, spacing, nspacing, interpolation=cv2.INTER_NEAREST):
    xspacing, yspacing, zspacing = spacing
    xnspacing, ynspacing, znspacing = nspacing

    xzoom, yzoom, zzoom = xspacing / xnspacing, yspacing / ynspacing, zspacing / znspacing

    data = __respace_iterate_second_axis( data,
                                          yzoom,
                                          xzoom,
                                          interpolation )

    data = __respace_iterate_zero_axis( data,
                                        zzoom,
                                        1.,
                                        interpolation )

    return data

def reshape(data, shape, nshape, interpolation=cv2.INTER_NEAREST):
    xsize, ysize, zsize = shape
    xnsize, ynsize, znsize = nshape

    xzoom, yzoom, zzoom = xnsize / xsize, ynsize / ysize, znsize / zsize

    data = __respace_iterate_second_axis( data,
                                          yzoom,
                                          xzoom,
                                          interpolation )

    data = __respace_iterate_zero_axis( data,
                                        zzoom,
                                        1.,
                                        interpolation )

    return data
