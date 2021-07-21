from scipy.ndimage import gaussian_filter
import numpy as np
import math
import os


def norm(a):
    return (a - a.min()) / (a.max() - a.min())


def define_dims(resolution):
    r"""
    Input arguments: resolution
    """
    II = 360 // resolution
    JJ = 180 // resolution
    return int(II), int(JJ)


def extract_region(mdt, lon_range, lat_range, central_lon=0, central_lat=0):
    res = mdt.shape[0] // 180

    px = ((lon_range[0] + central_lon) * res, (lon_range[1] + central_lon) * res)
    py = ((lat_range[0] + 90) * res, (lat_range[1] + 90) * res)

    return mdt[py[0]:py[1], px[0]:px[1]]


def create_coords(resolution, central_lon=0, rads=False):
    r"""
    Defines gloibal lon and lat, with lat shifted to
    midpoints and set between -90 and 90.
    """
    II, JJ = define_dims(resolution)
    longitude = np.array([resolution * (i + 0.5) - central_lon for i in range(II)])
    # quickfix
    # longitude[longitude>180] = longitude[longitude>180] - 360
    latitude = np.array([resolution * (j - 0.5) - 90.0 for j in range(JJ)])
    if rads:
        longitude = np.deg2rad(longitude)
        latitude = np.deg2rad(latitude)

    return longitude, latitude


def read_surface(filename, path=None, fortran=True, nans=True,
                 transpose=False, rotate=True):
    r"""Reshapes surface from 1d array into an array of
    (II, JJ) records.

    Ignores the header and footer of each record.

    Args:
        file (np.array): A .dat file containing a 1D array of floats
            respresenting input surface.

    Returns:
        np.array: data of size (II, JJ)
    """
    order = 'F' if fortran else 'C'

    if path is None:
        path = ""

    filepath = os.path.join(os.path.normpath(path), filename)
    fid = open(filepath, mode='rb')
    buffer = fid.read(4)
    size = np.frombuffer(buffer, dtype=np.int32)[0]
    shape = (int(math.sqrt(size//8)*2), int(math.sqrt(size//8)))
    fid.seek(0)

    # Loads Fortran array (CxR) or Python array (RxC)
    floats = np.array(np.frombuffer(fid.read(), dtype=np.float32), order=order)
    floats = floats[1:len(floats)-1]
    floats = np.reshape(floats, shape, order=order)

    if nans:
        floats[floats <= -1.7e7] = np.nan
    if transpose:
        return floats.T
    if rotate:
        return np.rot90(floats, 1)

    return floats
