from scipy.ndimage import gaussian_filter
import numpy as np
import math
import matplotlib.pyplot as plt


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


def rb_gaussian(II, JJ, lon_d, lat_d, data, mask, frad, fltr_type):
    print('data shape ', data.shape)
    # Hacky fix
    II = II + 1
    # new_lat_d = np.zeros((lat_d.shape[0] + 1))
    # new_lat_d[1:] = lat_d
    # new_lat_d[0] = lat_d[0] - (lat_d[1] - lat_d[0])
    # lat_d = np.append([lat_d[0] - (lat_d[1] - lat_d[0])], lat_d)
    # mask = np.append()
    lon_d = np.insert(lon_d, 0, lon_d[0] - (lon_d[1] - lon_d[0]))
    mask = np.insert(mask, 0, 1, axis=0)
    data = np.insert(data, 0, 0, axis=0)
    print(lat_d.shape, mask.shape)


    r = 6378136.3
    sigma = frad/np.sqrt(2.0*np.log(2.0))
    print('sigma: ', sigma)

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
        print('sd: ', sd)
        ang = 2.0*np.arcsin(0.5*sd/r)
        print('ang: ', ang)
        d = r*ang
        print('d: ', d)

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
        print('working on row ', j)

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

        for m in range(m1, m2):
            for n in range(II):
                x2 = r*np.cos(lat[m])*np.cos(lon[n])
                y2 = r*np.cos(lat[m])*np.sin(lon[n])
                z2 = r*np.sin(lat[m])

                sd = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
                ang = 2.0 * np.arcsin(0.5*sd/r)
                d = r * ang
                if fltr_type == 'box' and d <= frad:
                    fltr[n,m] = 1.0
                if fltr_type == 'gsn' and d <= 10.0 * sigma:
                    fltr[n,m] = np.exp(-0.5 * d**2/ sigma**2)/(np.sqrt(2*math.pi) *sigma)
                if fltr_type == 'tgn' and d <= 2*frad:
                    fltr[n,m] = np.exp(-0.5 * d**2/ sigma**2)/(np.sqrt(2*math.pi) *sigma)
                # Think the following may be the wrong way around
                if fltr_type == 'han' and d <= 2*frad:
                    fltr[n,m] = 0.54 + 0.46*np.cos(d* math.pi/(2*frad))
                if fltr_type == 'ham' and d <= 2*frad:
                    fltr[n,m] = 0.5 + 0.5*np.cos(d*math.pi/(2*frad))
        # plt.imshow(fltr)
        # plt.show()
        
        for i in range(II):
            # print('i,j: ', i, j)
            if mask[i,j]==0.0:

                # Translate the filter weights
                if i==0:
                    tfltr=fltr
                else:
                    tfltr[:i-1,:] = fltr[II-i+1:,:]
                    tfltr[i-1:,:] = fltr[:II-i+1,:]

                # Apply filter
                sm = 0.0
                for n in range(II):
                    for m in range(m1, m2):
                        if tfltr[n,m] !=0.0 and mask[n,m]==0.0:
                            sdata[i,j] = sdata[i,j] + tfltr[n,m]*data[n,m]
                            sm = sm+tfltr[n,m]

                # print(i, j, sm, np.sum(tfltr), np.sum(fltr))
                if sm!=0.0:
                    sdata[i,j] = sdata[i,j]/sm
                rsd[i,j] = data[i,j]-sdata[i,j]
        # plt.imshow(sdata)
        # plt.show()
    print('rb filter outputs: ', sdata.shape, rsd.shape)
    # remove first row j = 0
    return sdata[1:, :], rsd[1:, :]

def main():
    pass


if __name__ == "__main__":
    main()
