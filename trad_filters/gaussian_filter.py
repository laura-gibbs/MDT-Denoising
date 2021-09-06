from scipy.ndimage import gaussian_filter
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime


def apply_gaussian(arr, sigma=3, bd=-1.5):
    V = arr.copy()
    mask = np.ones_like(arr)
    mask[np.isnan(arr)] = np.nan
    V[np.isnan(arr)] = 0
    VV = gaussian_filter(V, sigma=sigma)

    W = 0*arr.copy() + 1
    W[np.isnan(arr)] = 0
    WW = gaussian_filter(W, sigma=sigma)

    arr = VV/WW * mask
    # arr[np.isnan(arr)] = bd

    return arr


def rb_filter(II, JJ, lon_d, lat_d, data, mask, frad, fltr_type):
    r = 6378136.3
    sigma = frad/np.sqrt(2.0*np.log(2.0))
    print('sigma: ', sigma)
    print(lon_d.shape, lat_d.shape)
    print('data shape ', data.shape)
    print('mask shape ', mask.shape)

    # Hacky fix
    II = II + 1
    lon_d = np.insert(lon_d, 0, lon_d[0] - (lon_d[1] - lon_d[0]))
    mask = np.insert(mask, 0, 1, axis=0)
    data = np.insert(data, 0, 0, axis=0)
    print(lat_d.shape, mask.shape)

    # Convert lons/lats to radians
    lon = lon_d*math.pi/180.0
    lat = lat_d*math.pi/180.0

    # Filter
    x1 = r*np.cos(lat[0])*np.cos(lon[0])
    y1 = r*np.cos(lat[0])*np.sin(lon[0])
    z1 = r*np.sin(lat[0])


    # Compute meridional radius of filter (assume same for all latitudes)
    ltrd = 1
    # What to put inside range:
    while True:
        x2 = r*np.cos(lat[ltrd])*np.cos(lon[0])
        y2 = r*np.cos(lat[ltrd])*np.sin(lon[0])
        z2 = r*np.sin(lat[ltrd])
        sd = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
        ang = 2.0*np.arcsin(0.5*sd/r)
        d = r*ang

        if fltr_type=='box' and d > frad:
            break
        elif fltr_type=='gsn' and d > 10.0*sigma:
            print('gaussian filter working')
            break
        elif fltr_type=='tgn' and d > 2*frad:
            break
        elif fltr_type=='han' and d > 2*frad:
            break
        elif fltr_type=='ham' and d > 2*frad:
            break
        ltrd = ltrd + 1
        
    # Chosen to define them as follows because they weren't defined
    sdata = np.zeros((II, JJ))
    rsd = np.zeros((II,JJ))

    print('ltrd: ', ltrd)
    for j in range(JJ):
        # print('working on row ', j)

        # Comput lat limits of filter window at lat j
        m1 = j - (ltrd - 1)
        if m1 < 0: 
            m1 = 0
        m2 = j + (ltrd - 1)
        if m2 >= JJ:
            m2 = JJ - 1
        
        fltr = np.zeros((II, JJ))
        tfltr = np.zeros((II, JJ))

        # Should indices be 1 or 0?
        x1 = r*np.cos(lat[j])*np.cos(lon[0])
        y1 = r*np.cos(lat[j])*np.sin(lon[0])
        z1 = r*np.sin(lat[j])

        # TIMING parts:
        filter_start = datetime.now()
        lat_test = np.zeros((II, JJ))
        lon_test = np.zeros((II, JJ))

        d = np.zeros((II, JJ))

        for m in range(m1, m2):
            for n in range(II):
                x2 = r*np.cos(lat[m])*np.cos(lon[n])
                y2 = r*np.cos(lat[m])*np.sin(lon[n])
                z2 = r*np.sin(lat[m])
                # lat_test[n, m] = lat[m]
                # lon_test[n, m] = lon[n]

                sd = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
                ang = 2.0 * np.arcsin(0.5*sd/r)
                # d = r * ang
                d[n, m] = r * ang
                # if fltr_type == 'box' and d <= frad:
                #     fltr[n,m] = 1.0
                # if fltr_type == 'gsn' and d <= 10.0 * sigma:
                #     fltr[n,m] = np.exp(-0.5 * d**2/ sigma**2)/(np.sqrt(2*math.pi) *sigma)
                # if fltr_type == 'tgn' and d <= 2*frad:
                #     fltr[n,m] = np.exp(-0.5 * d**2/ sigma**2)/(np.sqrt(2*math.pi) *sigma)
                # # Think the following may be the wrong way around
                # if fltr_type == 'han' and d <= 2*frad:
                #     fltr[n,m] = 0.54 + 0.46*np.cos(d* math.pi/(2*frad))
                # if fltr_type == 'ham' and d <= 2*frad:
                #     fltr[n,m] = 0.5 + 0.5*np.cos(d*math.pi/(2*frad))

        valid_mask = np.less_equal(d, 10.0 * sigma)
        out = np.exp(-0.5 * d ** 2 / sigma ** 2) / (np.sqrt(2 * math.pi) * sigma) * valid_mask
        fltr[:, m1:m2] = out[:, m1:m2]

        for i in range(II):
            # print('i,j: ', i, j)
            if mask[i,j]==0.0:

                # Translate the filter weights
                if i==0:
                    tfltr=fltr
                else:
                    tfltr[:i-1,:] = fltr[II-i+1:,:]
                    tfltr[i-1:,:] = fltr[:II-i+1,:]

                # # # Apply filter
                # # sm = 0.0

                # for n in range(II):
                #     for m in range(m1, m2):
                #         if tfltr[n,m] !=0.0 and mask[n,m]==0.0:
                # #             sdata[i,j] = sdata[i,j] + tfltr[n,m]*data[n,m]
                #             sm = sm+tfltr[n,m]
                
                # Compute land mask between m1 and m2 - land is zero here
                land_mask = np.logical_not(mask[:, m1:m2])

                # Compute convolution between filter and data, ignoring land pixels
                sdata[i, j] = np.sum(tfltr[:, m1:m2] * data[:, m1:m2] * land_mask)

                # Calculate sum of filter between m1 and m2, ignoring land pixels
                sm = np.sum(tfltr[:, m1:m2] * land_mask)


                # print(i, j, sm, np.sum(tfltr), np.sum(fltr))
                if sm!=0.0:
                    sdata[i,j] = sdata[i,j]/sm
                #rsd[i,j] = data[i,j]-sdata[i,j]

        # plt.imshow(sdata)
        # plt.show()
    print('rb filter outputs: ', sdata.shape, rsd.shape)
    # remove first row j = 0
    return sdata[1:, :]

def main():
    pass


if __name__ == "__main__":
    main()
