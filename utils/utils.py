import numpy as np


def define_dims(resolution):
    r"""
    Input arguments: resolution
    """
    II = 360 // resolution
    JJ = 180 // resolution
    return int(II), int(JJ)


def create_coords(resolution, rads=False):
    r"""
    Defines gloibal lon and lat, with lat shifted to
    midpoints and set between -90 and 90.
    """
    II, JJ = define_dims(resolution)
    longitude = np.array([resolution * (i - 0.5) for i in range(II)])
    latitude = np.array([resolution * (j - 0.5) - 90.0 for j in range(JJ)])
    if rads:
        longitude = np.deg2rad(longitude)
        latitude = np.deg2rad(latitude)

    return longitude, latitude


def get_spatial_params(x, y, w=128, h=128, resolution=0.25):
    coord_grid = create_coords(resolution)
    lons = coord_grid[0][x:x+w]
    lats = coord_grid[1][int(180/resolution)-y:int(180/resolution)-(y+h):-1]
    x0, x1 = x * resolution, (x + w) * resolution
    y0, y1 = 90 - y * resolution, 90 - (y + h) * resolution
    if x0 > 180:
        x0 -= 360
        x1 -= 360
    extent = x0, x1, y0, y1
    return lons, lats, extent


def get_region_coords(resolution, region_size_d=32):
    r'''
    resolution (float): degree resolution
    region_size_d (int): region size in degrees
    '''
    x_coords = []
    y_coords = []
    region_size = region_size_d / resolution
    # For threshold 64 North and South
    y0 = (90 - 64) / resolution
    y1 = (180 - 26) / resolution
    x0 = 0
    x1 = 360 / resolution
    y = y0
    for i in range((y1 - y0) // region_size):
        y = i * region_size + y0
        for j in range((x1 - x0) // region_size):
            x = j * region_size
            x_coords.append(x)
            y_coords.append(y)
    return x_coords, y_coords


def bound_arr(arr, lower_bd, upper_bd):
    arr[np.isnan(arr)] = lower_bd
    arr[arr < lower_bd] = lower_bd
    arr[arr > upper_bd] = upper_bd
    return arr

