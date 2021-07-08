import torch
from denoising.models import ConvAutoencoder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from denoising.utils import apply_gaussian, create_coords, read_surface
from denoising.data import CAEDataset
from utils.utils import get_spatial_params
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature import GSHHSFeature
from trad_filters.gaussian_filter import rb_gaussian, apply_gaussian

from skimage.metrics import structural_similarity as ssim
import cv2


def mutual_information(hgram):
    """ Mutual information for joint histogram
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def discretize(target, output):
    disc_output = output.copy()
    disc_target = target.copy()
    dmax = np.max(disc_output)
    dmin = np.min(disc_output)
    disc_output[disc_output < (dmax - dmin)/3] = dmin
    disc_output[np.logical_and(disc_output > (dmax - dmin)/3, disc_output < (dmax - dmin)/3*2 )] = (dmax - dmin)/2
    disc_output[disc_output > (dmax - dmin)/3*2] = dmax

    disc_target[disc_target < (dmax - dmin)/3] = dmin
    disc_target[np.logical_and(disc_target > (dmax - dmin)/3, disc_target < (dmax - dmin)/3*2 )] = (dmax - dmin)/2
    disc_target[disc_target > (dmax - dmin)/3*2] = dmax

    # target_norm = (target - np.min(output))/(np.max(output) - np.min(output))
    residual = output[0,0] - target[0]
    disc_residual = residual.copy()
    dmax = np.max(residual)
    dmin = np.min(residual)
    # print(dmin, dmax)
    disc_residual[np.logical_and(disc_residual > (dmax - dmin)/4, disc_residual < (dmax - dmin)/4*3 )] = (dmax - dmin)/2
    disc_residual[disc_residual < (dmax - dmin)/4] = dmin
    disc_residual[disc_residual > (dmax - dmin)/4*3] = dmax



class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



def load_batch(dataset, batch_size, batch_intersect=0):
    images = []
    paths = []
    targets = []
    for i in range(batch_intersect, batch_intersect+batch_size):
        image, target = dataset[i]
        images.append(image)
        targets.append(target)
        paths.append(dataset.paths[i])
    images = torch.stack(images)
    if dataset.testing:
        return images, paths
    else:
        targets = torch.stack(targets)
        return images, targets, paths


def detach(input):
    return input.detach().cpu().numpy()


def detach_tensors(tensors):
    tensors = [detach(tensor) for tensor in tensors]
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def land_false(arr):
    # For multiplying
    return arr != 0


def land_true(arr): #land_mask
    # For indexing
    return arr == 0


def get_bounds(mdt=False):
    if mdt:
        vmin = -1.5
        vmax = 1.5
    else:
        vmin = 0
        vmax = 1.5
    return vmin, vmax


def get_cbar(fig, im, cbar_pos, mdt=False, rmse=False, orientation='vertical', labelsize=9):
    r"""
    Args:
        cbar_pos (list): left_pos, bottom_pos, cbarwidth, cbarheight
    """
    vmin, vmax, cticks = get_plot_params(mdt, rmse)
    cbar_ax = fig.add_axes(cbar_pos)
    cbar_ax.tick_params(labelsize=labelsize)
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.linspace(vmin, vmax, num=cticks), orientation=orientation)
    return cbar


def compute_avg_residual(arrs, reference):
    r"""
    Args:
        arrs(list): list of mdt/cs arrays
        reference(arr): single reference array 
    """
    residual = arrs - reference
    avg_residual = np.mean(arrs - reference, axis=(0,1))
    return avg_residual


def compute_avg_rmsd(arrs, reference, **kwargs):
    r"""
    Args:
        arrs(list):
        reference(arr)
    """
    residuals = arrs - reference
    rmsds = np.array([compute_rmsd(residual[0], **kwargs) for residual in residuals])
    # for i in range(rmsds.shape[0]):
    #     plt.imshow(rmsds[i, 0])
    #     plt.show()
    # print(rmsds.shape)
    # print('here', np.mean(rmsds, axis=0).shape)
    return np.mean(rmsds, axis=0)




def compute_rmsd(residual, hw_size=5, mask=None, res=4):
    r"""
    Args:
        hw_size (integer): half width size of window
    """
    squared_diff = residual**2

    # Get first dimension? and divide 360 by it to get degree resolution
    # might not work if they're torch tensors!! 
    # print(residual.shape[0])
    # res = residual.shape[0]/360  # only true if it's a global map
    rmsd = np.zeros_like(residual)
    hw = int(hw_size * res) # kernel size in pixels
    # print(hw)
    # Add reflected padding of size hw
    squared_diff = cv2.copyMakeBorder(squared_diff, hw, hw, hw, hw, borderType=cv2.BORDER_REFLECT)
    # Convolve window of sixe hw across image and average
    for i in range(residual.shape[0]):
        for j in range(residual.shape[1]):
        # centre of the moving window index [i,j]
            window = squared_diff[i:i+hw*2,j:j+hw*2]
            # print(np.shape(window))
            # print(np.shape(rmsd))
            rmsd[i,j] = np.sqrt(np.mean(window))
    if mask is not None:
        return rmsd * mask
    return rmsd


def plot_region(tensor, ax, lons, lats, extent, crs=ccrs.PlateCarree(), **plot_kwargs):
    if 'cmap' not in plot_kwargs:
        plot_kwargs['cmap']='turbo'
    x0, x1, y0, y1 = extent
    im = ax.pcolormesh(lons, lats, tensor, transform=crs, **plot_kwargs)
    ax.set_extent((x0, x1, y0, y1), crs=crs)
    # longitude[longitude>180] = longitude[longitude>180] - 360
    ax.set_xticks(np.linspace(x1, x0, 5), crs=crs)
    ax.set_yticks(np.linspace(y0, y1, 5), crs=crs)
    lat_formatter = LatitudeFormatter()
    lon_formatter = LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.add_feature(GSHHSFeature(scale='intermediate', facecolor='lightgrey', linewidth=0.2))
    return im


def get_plot_params(mdt=False, rmse=False, residual=False):
    if mdt:
        if rmse:
            vmin = 0
            vmax = 0.2
            cticks = 7
        else:
            vmin = -1.5
            vmax = 1.5
            cticks = 7
    else:
        if rmse:
            vmin = 0
            vmax = 0.15
            cticks = 7
        else:
            vmin = 0
            vmax = 1.5 # Change back to 1.5
            cticks = 7
    if residual:
        vmin = -0.6
        vmax = 0.6
        cticks = 11
    return vmin, vmax, cticks


def create_subplot(data, regions, mdt=False, rmse=False, residual=False, cols=3, crs=ccrs.PlateCarree(), titles=None, big_title=None):
    vmin, vmax, cticks = get_plot_params(mdt, rmse, residual)
    if rmse:
        cmap = 'jet' #'hot_r'
    elif residual:
        cmap='bwr'#, norm=MidpointNormalize(midpoint=0.)
    else:
        cmap = 'turbo'
    fig, axes  = plt.subplots(len(data) // cols, cols, figsize=(25, 10), subplot_kw={'projection': crs})#, squeeze=False)
    if titles is None:
        titles = [''] * len(data)
    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            # split_path = paths[j][:len(paths[j])-4].split('_')
            # x, y = int(split_path[-2]), int(split_path[-1])
            x, y = regions[i * len(axrow) + j]
            # w, h = tensor[j, 0].shape
            lons, lats, extent = get_spatial_params(x, y)
            im = plot_region(data[i * len(axrow) + j], ax, lons, lats, extent, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(titles[i * len(axrow) + j])
    cbarheight = 0.75
    bottom_pos = (1 - cbarheight)/2 - 0.005
    cbarwidth = 0.01
    left_pos = 0.92 # (should be half of cbarwidth to be center-aligned if orien=horiz)
    cbar_ax = fig.add_axes([left_pos, bottom_pos, cbarwidth, cbarheight])
    cbar_ax.tick_params(labelsize=9)
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.linspace(vmin, vmax, num=cticks))#, orientation='horizontal')
    if big_title is not None:
        fig.suptitle(big_title, fontsize=16)
    # fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
    plt.close()




def main():
    mdt = False
    n_epochs = 160
    crs = ccrs.PlateCarree()

    if mdt:
        var = 'mdt'
    else:
        var = 'cs'

    model = ConvAutoencoder()
    # # model.load_state_dict(torch.load(f'./models/exp_epochs_{var}/{n_epochs}e_{var}_model_cdae.pth'))
    model.eval()

    dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/small_testing_{var}', quilt_dir=f'./quilting/DCGAN_{var}', mdt=mdt)
    # dataset = CAEDataset(region_dir=f'../a_mdt_data/testing_geodetic_data/{var}', quilt_dir=None, mdt=mdt)
    t_dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/small_testing_{var}', quilt_dir=None, mdt=mdt)
    # t_dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/orca_{var}_regions', quilt_dir=None, mdt=mdt)
    # t_dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/cls18_{var}_regions', quilt_dir=None, mdt=mdt)

    batch_size = 6
    batch_intersect = 0
    x_coords = [0, 128, 128, 256, 256, 384, 384, 512]#, 512, 768, 768, 768, 768, 896, 896, 896] # 640, 640, 640, 640
    y_coords = [488, 360, 488, 360, 488, 360, 488, 232]#, 488, 104, 232, 360, 488, 232, 360, 488] # 104, 232, 360, 488

    x, y = 0, 488

    g_images = dataset.get_regions(x, y)
    g_images = torch.stack(g_images)
    target = t_dataset.get_regions(x,y)[0]
    g_images, target = detach_tensors([g_images, target])
    mask = land_false(g_images)[0]
    target = target * mask
    g_images = g_images * mask


    # Gaussian filtered MDT: test for one filter radius
    # -------------------------
    # sigma = ((320 * 4) / 111) / 2.355
    # gauss_images = [apply_gaussian(image, mdt, sigma) for image in g_images]
    # gauss_images = np.array(gauss_images)
    # print(gauss_images.shape)
    

    # # Using RB's filter: test for one filter radius
    # # ---------------------------
    # # g_images = g_images[:2,:,:,:]
    # print(g_images.shape)
    # print(type(g_images))
    # g_mask = g_images[0] == 0
    # lons, lats, _ = get_spatial_params(x, y)
    # II, JJ = 128, 128
    # print('g_images ', g_images[0].shape, g_mask.shape)
    # gauss_images = [rb_gaussian(II, JJ, lons, lats, image.squeeze(0), g_mask[0], 50000, 'gsn')[0] for image in g_images]
    # print(type(gauss_images))
    # gauss_images = np.array(gauss_images)
    # gauss_images = np.expand_dims(gauss_images, axis=1)
    # print(gauss_images.shape)


    

    
    # Calculate Gaussian filtered MDT using rb_gaussian across multiple filter radii
    # --------------------------------------------
    sigmas = [25000, 50000, 100000, 200000]
    II, JJ = 128, 128
    gauss_rmsds = []
    gauss_avg_rmses = []
    for sigma in sigmas:
        print('Working for sigma: ', sigma)
        g_mask = g_images[0] == 0
        lons, lats, _ = get_spatial_params(x, y)
        gauss_images = [rb_gaussian(II, JJ, lons, lats, image.squeeze(0), g_mask[0], sigma, 'gsn')[0] for image in g_images]
        gauss_images = np.array(gauss_images)
        gauss_images = np.expand_dims(gauss_images, axis=1)

        # For line graph
        gauss_avg_rmses.append(np.sqrt(np.mean(gauss_images - target) **2))
        
        # For pixelwise plot
        gauss_rmsd = compute_avg_rmsd(gauss_images, target, hw_size=5, mask=mask)
        gauss_rmsds.append(gauss_rmsd)
        # print('gauss rmsd shape: ', gauss_rmsd.shape)


    # Calculate RMSD: test for one filter radius
    # -----------------------
    gauss_rmsd = compute_avg_rmsd(gauss_images, target, hw_size=5, mask=mask)
    plt.imshow(gauss_rmsd[0], cmap='jet', vmax=0.2)
    plt.show()


    # Providing first images for following subplots (Gaussian filtered for RMSD and Unfiltered for Residuals)
    # -----------------------------------------------
    print('gauss_rmsd shape', gauss_rmsd.shape)
    rmsds = [gauss_rmsd[0]]
    avg_residuals = [compute_avg_residual(g_images, target)]
    print(len(avg_residuals))
    residuals = [(g_images[0] - target).squeeze(0)]


    # Computing RMSD for region (x,y) across different number of epochs
    # --------------------------------------------------
    r = 2
    a = 5
    epoch_inc = 25
    epochs = [a * r**i for i in range(6)]
    # epochs = list(range(epoch_inc, 175, epoch_inc))
    avg_rmses = []
    print(epochs)
    for epoch in epochs:
        g_images = dataset.get_regions(x, y)
        g_images = torch.stack(g_images)
        target = t_dataset.get_regions(x,y)[0]
        model.load_state_dict(torch.load(f'./models/multiscale_loss_{var}/{epoch}e_{var}_model_cdae.pth'))
        # model.load_state_dict(torch.load(f'./models/{epoch_inc}epochs_{var}/{epoch}e_{var}_model_cdae.pth'))
        g_outputs = model(g_images)
        g_images, g_outputs, target = detach_tensors([g_images, g_outputs, target])
        mask = land_false(g_images)[0]
        target = target * mask
        g_images = g_images * mask
        g_outputs = g_outputs * mask

        avg_residual = compute_avg_residual(g_outputs, target)
        avg_residuals.append(avg_residual)
        residual = (g_outputs[0] - target)
        residuals.append(residual[0])
        
        # For line graph
        avg_rmses.append(np.sqrt(np.mean(avg_residual - target) **2))

        # rmsd = compute_rmsd(avg_residual, hw_size=5, mask=mask)
        rmsd = compute_avg_rmsd(g_outputs, target, hw_size=5, mask=mask)
        rmsds.append(rmsd[0])
    print(np.shape(g_outputs))

    # create_subplot(g_outputs.squeeze(1), [[x, y]] * batch_size, mdt=mdt, rmse=False, residual=False, titles=[f'{epoch} epochs' for epoch in epochs])#, big_title='RMSD between network output and cls - trained over a different number of epochs')    
    # create_subplot(g_images.squeeze(1), [[x, y]] * batch_size, mdt=mdt, rmse=False, residual=False, titles=[f'{epoch} epochs' for epoch in epochs])
    # print(len(avg_residuals))
    create_subplot(rmsds, [[x, y]] * batch_size, mdt=mdt, rmse=True, titles=['Gaussian Filtered (320km HW)'] + [f'{epoch} epochs' for epoch in epochs], big_title=f'RMSD between network output and target - trained over a different number of epochs ({var} trained)')    
    print(np.shape(residuals))
    create_subplot(residuals, [[x, y]] * batch_size, mdt=mdt, rmse=False, residual=True, titles=['Unfiltered'] + [f'{epoch} epochs' for epoch in epochs], big_title=f'Residual difference (Output - Target) trained over a different number of epochs ({var} trained)')
    

    # Plot line graph of epochs against Gaussian RMSD
    # ----------------------------------------------
    avg_rmse = np.sqrt(np.mean((g_images - target) ** 2))
    plt.plot(sigmas, gauss_avg_rmses)
    plt.axhline(y=avg_rmse, color='r', linestyle='dashed')
    plt.legend(["Gaussian filter output", "CDAE output at 200 epochs"], loc ="upper right")
    plt.xlabel('Gaussian Filter Half-weight Radius (km)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    # plt.ylim(bottom=0.14, top=0.23)
    plt.title('RMSE of Different Filtering Methods Against NEMO Data')
    plt.show()
    plt.close()


    fig, ax = plt.subplots(1,4)
    ax[0].imshow(g_images[0,0])#, vmin=-0.6, vmax=0.6)
    ax[1].imshow(g_outputs[0,0])#, vmin=-0.6, vmax=0.6)
    ax[2].imshow(target[0])#, vmin=-0.6, vmax=0.6)
    ax[3].imshow(residuals[-1])#, vmin=-0.6, vmax=0.6)
    plt.show()
    plt.close()


    # # ------------------------------------------
    # # Calculate Standard Gaussian Filtered Geodetic MDTs and RMSE/SSIM between Gauss vs NEMO and Model vs NEMO
    # # FWH_pixels  = 2.355 * sigma
    # # FWH_km = (FWH_pixels / 4) * 111
    # kms = np.arange(25, 501, 25)
    # sigmas = ((kms * 4) / 111) / 2.355
    # gauss_avg_rmses = []
    # gauss_avg_ssim = []
    # for sigma in sigmas:
    #     gauss_images = [apply_gaussian(image, mdt, sigma) for image in g_images]
    #     gauss_images = np.array(gauss_images)
    #     gauss_avg_rmses.append(np.sqrt(np.mean((gauss_images - nemo) ** 2)))
    #     gauss_avg_ssim.append(np.mean([ssim(nemo_image[0], gauss_image[0]) for nemo_image, gauss_image in zip(nemo, gauss_images)]))
    # print('gauss rmse:', gauss_avg_rmses)
    # print('gauss ssim:', gauss_avg_ssim)
    # plt.imshow(gauss_images[0, 0])
    # plt.close()
    
    
    # Alternative avg rmsds method? Uses compute_rmsd instead of compute_avg_rmsd?
    # Uses x_coords so it loops through the different regions
    avg_rmses = []
    rmsds = []
    for x, y in zip(x_coords, y_coords):
        g_images = dataset.get_regions(x, y)
        g_images = torch.stack(g_images)
        g_outputs = model(g_images)
        target = t_dataset.get_regions(x,y)[0]
        g_images, g_outputs, target = detach_tensors([g_images, g_outputs, target])
        mask = land_false(g_images)[0]
        target = target * mask
        g_images = g_images * mask
        g_outputs = g_outputs * mask
        avg_residual = compute_avg_residual(g_outputs, target)
    
        # # -------
        # # Felix's code to calculate a second residual and plot both side by side
        # # -------
        # target2 = t_dataset.get_regions(x,y)[0]
        # avg_residual2 = compute_avg_residual(g_outputs, target2.numpy() * mask)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(avg_residual)
        # ax[1].imshow(avg_residual2)
        # plt.show()

        rmsd = compute_rmsd(avg_residual, hw_size=3, mask=mask)
        rmsds.append(rmsd[0])
        lons, lats, extent = get_spatial_params(x, y)
        # fig, ax = plt.subplots(1, 1, subplot_kw={'projection':ccrs.PlateCarree()})
        # im = plot_region(rmsd[0], ax, lons, lats, extent, cmap='hot')
        # plt.imshow(rmsd[0], vmin=0.3, vmax=1)
        # plt.show()
        avg_rmse = np.sqrt(np.mean((g_outputs - target) ** 2))
        avg_rmses.append(avg_rmse)
  




    # plt.plot(kms, gauss_avg_ssim)
    # plt.axhline(y=avg_ssim, color='r', linestyle='dashed')
    # plt.legend(["Gaussian filter output", "CDAE output at 200 epochs"], loc ="lower right")
    # plt.xlabel('Gaussian Filter Half-weight Radius (km)', fontsize=12)
    # plt.ylabel('SSIM', fontsize=12)
    # # plt.ylim(bottom=0.05, top=0.8)
    # plt.title('SSIM of Different Filtering Methods With NEMO Data')
    # plt.show()
    # plt.close()


    # fig, axes  = plt.subplots(2, batch_size, figsize=(25, 10), subplot_kw={'projection': crs}, squeeze=False)
    # if mdt:
    #     g_images[lmask] = vmin
    #     g_outputs[lmask] = vmin

    # for i, (axrow, tensor) in enumerate(zip(axes, [g_images, g_outputs])):
    #     for j, ax in enumerate(axrow):
    #         split_path = g_paths[j][:len(g_paths[j])-4].split('_')
    #         x, y = int(split_path[-2]), int(split_path[-1])
    #         # w, h = tensor[j, 0].shape
    #         lons, lats, extent = get_spatial_params(x, y)
    #         im = plot_region(tensor[j, 0], ax, lons, lats, extent, vmin=vmin, vmax=vmax)
    # cbarheight = 0.75
    # bottom_pos = (1 - cbarheight)/2 - 0.005
    # cbarwidth = 0.01
    # left_pos = 0.92 # (should be half of cbarwidth to be center-aligned if orien=horiz)
    # cbar = get_cbar(fig, im, [left_pos, bottom_pos, cbarwidth, cbarheight], mdt)
    # plt.show()
    # plt.close()

    # -----------------------------------------
    # Load CMIP images with added synthetic noise
    images, targets, paths = load_batch(dataset, batch_size)
    outputs = model(images)
    images, outputs, targets = detach_tensors([images, outputs, targets])
    mask = land_false(targets)
    lmask = land_true(targets)

    if not mdt:
        outputs = outputs * mask

    residual = (targets - outputs) * mask


    # -------------------------------------------
    # Plot input, target, outputs for multiple regions
    fig, axes  = plt.subplots(3, batch_size, figsize=(25, 10), subplot_kw={'projection': crs})
    if mdt:
        images[lmask] = vmin
        outputs[lmask] = vmin
        targets[lmask] = vmin

    for i, (axrow, tensor) in enumerate(zip(axes, [images, outputs, targets])):
        for j, ax in enumerate(axrow):
            split_path = paths[j][:len(paths[j])-4].split('_')
            x, y = int(split_path[-2]), int(split_path[-1])
            # w, h = tensor[j, 0].shape
            lons, lats, extent = get_spatial_params(x, y)
            im = plot_region(tensor[j, 0], ax, lons, lats, extent, vmin=vmin, vmax=vmax)
    cbarheight = 0.75
    bottom_pos = (1 - cbarheight)/2 - 0.005
    cbarwidth = 0.01
    left_pos = 0.92 # (should be half of cbarwidth to be center-aligned if orien=horiz)
    cbar = get_cbar(fig, im, [left_pos, bottom_pos, cbarwidth, cbarheight], mdt)
    # plt.show()
    # plt.close()


    # Plot element wise RMSE across multiple regions
    rvmin = 0
    rvmax = 1.
    c_ticks = 11
    # if not mdt:
    #     rvmax = 0.2
    #     c_ticks = 11
    # else:
    #     rvmax = 0.35
    #     c_ticks = 8

    rmses = []
    for x, y in zip(x_coords, y_coords):
        regions = dataset.get_regions(x, y)
        target = nemo.get_regions(x, y)[0]
        regions = torch.stack(regions)
        output_regions = model(regions)
        target = target.detach().cpu().numpy()
        output_regions = output_regions.detach().cpu().numpy()
        mask = target != 0

        rmse = np.sqrt(np.mean((output_regions - target)**2, axis=0))
        rmse = rmse * mask
        rmses.append(rmse[0])
    fig, axes = plt.subplots(4, 4, subplot_kw={'projection': crs})
    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            index = i * 4 + j
            x, y = x_coords[index], y_coords[index]
            lons, lats, extent = get_spatial_params(x, y)
            print(rmses[index].shape)
            r_plot = plot_region(rmses[index], ax, lons, lats, extent,
                cmap='jet', vmin=rvmin, vmax=rvmax

            )
    cbarheight = 0.75
    bottom_pos = (1 - cbarheight)/2 - 0.005
    cbarwidth = 0.01
    left_pos = 0.92 # (should be half of cbarwidth to be center-aligned if orien=horiz)
    cax = fig.add_axes([left_pos, bottom_pos, cbarwidth, cbarheight])
    cax.tick_params(labelsize=9)
    cbar = fig.colorbar(r_plot, cax=cax, ticks=np.linspace(rvmin, rvmax, num=c_ticks))#, orientation='horizontal')
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()




# # Plotting a signle input, output, target, residual
# ax[0].imshow(images, cmap='turbo', vmin=vmin, vmax=vmax)
# ax[1].imshow(output, cmap='turbo', vmin=vmin, vmax=vmax)
# ax[2].imshow(targets, cmap='turbo', vmin=vmin, vmax=vmax)
# ax[3].imshow(residual, cmap='seismic', norm=MidpointNormalize(midpoint=0.))
# ax[0].title.set_text('Input')
# ax[1].title.set_text('Output')
# ax[2].title.set_text('Target')
# ax[3].title.set_text('Residual: (Output - Target)')
# plt.show()
# plt.close()

# # Plotting the signal
# fig, axes = plt.subplots(1, 3)
# axes[0].hist(image.ravel(), bins=20)
# axes[0].set_title('Input histogram')
# axes[1].hist(target.ravel(), bins=20)
# axes[1].set_title('Target histogram')
# axes[2].hist(output.ravel(), bins=20)
# axes[2].set_title('Output histogram')
# plt.show()

# # Scatter Diagram CorrCoef
# plt.plot(image.ravel(), target.ravel(), '.')
# plt.xlabel('Input signal')
# plt.ylabel('Target signal')
# plt.title('Input vs Target signal')
# np.corrcoef(image.ravel(), target.ravel())[0, 1]
# plt.show()
# plt.close()

# Plotting line of best fit
# m, b = np.polyfit(image, target, 1)
# plt.plot(image, m*image + b)
# plt.show()

# # 2D Histograms
# hist_2d, x_edges, y_edges = np.histogram2d(
#     image.ravel(),
#     target.ravel(),
#     bins=20)
# plt.imshow(hist_2d.T, origin='lower')
# plt.xlabel('Input')
# plt.ylabel('Target')
# plt.show()
# plt.close()

# # Log 2D Histograms
# hist_2d_log = np.zeros(hist_2d.shape)
# non_zeros = hist_2d != 0
# hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
# plt.imshow(hist_2d_log.T, origin='lower')
# plt.xlabel('Input')
# plt.ylabel('Target')
# plt.show()
# plt.close()


# # Compute Mutual Information
# print(mutual_information(hist_2d))

# # Discretize Plotting
# fig, ax  = plt.subplots(2,3)
# ax[0][0].imshow(image[0], cmap='turbo', vmin=vmin, vmax=vmax)
# ax[0][1].imshow(output[0,0], cmap='turbo', vmin=vmin, vmax=vmax)
# ax[0][2].imshow(target[0], cmap='turbo', vmin=vmin, vmax=vmax)
# ax[1][0].imshow(disc_output[0,0])
# ax[1][1].imshow(disc_target[0])
# ax[1][2].imshow(disc_output[0,0] - disc_target[0])
# ax[0][0].title.set_text('Input')
# ax[0][1].title.set_text('Output')
# ax[0][2].title.set_text('Target')
# ax[1][0].title.set_text('Discretized Output')
# ax[1][1].title.set_text('Discretized Target')
# ax[1][2].title.set_text('Disc. Output - Disc. Target') 
# plt.show()


# # Loading single regions for Gaussian filter plots
# target_img = np.load('../a_mdt_data/HR_model_data/training_regions/CNRM-CM6-1-HR_r1i1p1f2_gn_2001_cs_0_488.npy')
# quilt = Image.open('quilting/DCGAN_32deg/cut-b32-n5_2.png')
# quilt = quilt.convert(mode='L')
# quilt = np.array(quilt).astype(np.float32)
# mask = target_img != 0
# quilt = (quilt - np.nanmin(quilt)) / (np.nanmax(quilt) - np.nanmin(quilt))
# img = target_img + .3* quilt * mask

# # Gaussian
# mdt_img = np.load('../a_mdt_data/HR_model_data/training_regions/CNRM-CM6-1-HR_r1i1p1f2_gn_2001_cs_0_488.npy')
# gauss_img = apply_gaussian(img) * mask

# img = torch.tensor(img)
# print(img.shape)
# img = torch.unsqueeze(img, 0)
# img = torch.unsqueeze(img, 0)
# print(img.shape)

# # batch x channel x width x height
# output = model(img)
# # output = output.view(1, 1, 128, 128)
# output = output.detach().cpu().numpy()
# output = np.squeeze(output, 0)
# output = np.squeeze(output, 0)
# output = output * mask




