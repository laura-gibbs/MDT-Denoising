from random import gauss
import torch
from denoising.models import ConvAutoencoder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from denoising.utils import norm, create_coords, read_surface
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
    avg_residual = np.mean(arrs - reference, axis=0)
    return avg_residual


def compute_avg_rmsd(arrs, reference, **kwargs):
    r"""
    Args:
        arrs(list):
        reference(arr)
    """
    residuals = arrs - reference
    rmsds = np.array([compute_rmsd(residual, **kwargs) for residual in residuals])
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


def create_subplot(data, regions, mdt=False, rmse=False, residual=False, cols=3, crs=ccrs.PlateCarree(), titles=None, big_title=None, cmaps=None, ax_labels=None):
    vmin, vmax, cticks = get_plot_params(mdt, rmse, residual)
    if cmaps is None:
        if rmse:
            cmap = 'jet' #'hot_r'
        elif residual:
            cmap='bwr'#, norm=MidpointNormalize(midpoint=0.)
        else:
            cmap = 'turbo'
        cmaps = [cmap] * len(data)
    fig, axes  = plt.subplots(len(data) // cols, cols, figsize=(25, 10), subplot_kw={'projection': crs}, squeeze=False)
    if titles is None:
        titles = [''] * len(data)
    for i, axrow in enumerate(axes):
        for j, ax in enumerate(axrow):
            # split_path = paths[j][:len(paths[j])-4].split('_')
            # x, y = int(split_path[-2]), int(split_path[-1])
            x, y = regions[i * len(axrow) + j]
            # w, h = tensor[j, 0].shape
            lons, lats, extent = get_spatial_params(x, y)
            im = plot_region(data[i * len(axrow) + j], ax, lons, lats, extent, vmin=vmin, vmax=vmax, cmap=cmaps[i * len(axrow) + j])
            ax.set_title(titles[i * len(axrow) + j])
            if ax_labels is not None:
                ax.set_ylabel(ax_labels[i], fontsize=9)
    cbarheight = 0.75
    bottom_pos = (1 - cbarheight)/2 - 0.005
    cbarwidth = 0.01
    left_pos = 0.92 # (should be half of cbarwidth to be center-aligned if orien=horiz)
    cbar_ax = fig.add_axes([left_pos, bottom_pos, cbarwidth, cbarheight])
    cbar_ax.tick_params(labelsize=9)
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=np.linspace(vmin, vmax, num=cticks))#, orientation='horizontal')
    if big_title is not None:
        fig.suptitle(big_title, fontsize=12)
    fig.subplots_adjust(top=0.88)
    plt.tight_layout()
    plt.show()
    plt.close()


def calculate_rmse(x, y, normalise=False):
    if normalise:
        x, y = norm(x), norm(y)
    return np.sqrt(np.mean((x - y)**2))
    


# def standard_gaussian():
    # # Calculate Standard Gaussian Filtered Geodetic MDTs and RMSE/SSIM between Gauss vs NEMO and Model vs NEMO
    # # FWH_pixels  = 2.355 * sigma
    # # FWH_km = (FWH_pixels / 4) * 111
    # kms = np.arange(25, 501, 25)
    # sigmas = ((kms * 4) / 111) / 2.355
    # gauss_avg_rmses = []
    # # gauss_avg_ssim = []
    # for sigma in sigmas:
    #     gauss_images = [apply_gaussian(image, sigma) for image in g_images]
    #     gauss_images = np.array(gauss_images)
    #     gauss_avg_rmses.append(calculate_rmse(gauss_images, target[:, 0]))
    #     gauss_filtered.append(gauss_images[1])
    #     # gauss_avg_ssim.append(np.mean([ssim(nemo_image[0], gauss_image[0]) for nemo_image, gauss_image in zip(nemo, gauss_images)]))



def main():
    mdt = False
    crs = ccrs.PlateCarree()

    if mdt:
        var = 'mdt'
    else:
        var = 'cs'

    model = ConvAutoencoder()
    model.eval()

    batch_size = 6
    batch_intersect = 0
    x_coords = [0, 128, 128, 256, 256, 384, 384, 512]#, 512, 768, 768, 768, 768, 896, 896, 896] # 640, 640, 640, 640
    y_coords = [488, 360, 488, 360, 488, 360, 488, 232]#, 488, 104, 232, 360, 488, 232, 360, 488] # 104, 232, 360, 488

    # Gulf Stream
    # x, y = 1088, 168
    # Agulhas 
    # x, y = 32, 456
    n_epochs = 160
    x, y = 0, 488

    # 'orca' or 'cls'
    ref_var = 'orca'

    # Only one region needed as reference for geodetic data
    # -----------------------------------------------
    geodetic = False
    if geodetic:
        dataset = CAEDataset(region_dir=f'../a_mdt_data/new_testing_geodetic_data/{var}', quilt_dir=None, mdt=mdt)
        t_dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/new_{ref_var}_{var}_regions', quilt_dir=None, mdt=mdt)
        network_images = dataset.get_regions(x, y)
        network_images = torch.stack(network_images)
        target = t_dataset.get_regions(x,y)[0]
    else:
        dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/new_{ref_var}_{var}_regions', quilt_dir=f'./quilting/DCGAN_{var}', mdt=mdt)
        t_dataset = CAEDataset(region_dir=f'../a_mdt_data/HR_model_data/new_{ref_var}_{var}_regions', quilt_dir=None, mdt=mdt) #small_{var}_testing
        network_images = dataset.get_regions(x, y)
        network_images = torch.stack(network_images)
        target = t_dataset.get_regions(x,y)
        target = torch.stack(target)
        target = torch.squeeze(target, dim=1)
    
    images, target = detach_tensors([network_images, target])
    images = np.squeeze(images, axis=1) 
    
    mask = land_false(images)[0]
    target = target * mask
    images = images * mask
    plt.imshow(target[0], cmap='turbo')
    plt.show()

    gauss_filtered = []

    
    # Calculate Gaussian filtered MDT using rb_gaussian across multiple filter radii
    # --------------------------------------------
    sigmas = np.arange(10000, 150001, 10000)
    # sigmas = [10000] * 14 + [50000]
    II, JJ = 128, 128
    gauss_rmsds = []
    gauss_avg_rmsds = []
    for sigma in sigmas:
        print('Working for sigma: ', sigma)
        g_mask = images[0] == 0
        lons, lats, _ = get_spatial_params(x, y)
        gauss_images = [rb_gaussian(II, JJ, lons, lats, image, g_mask, sigma, 'gsn') for image in images]
        gauss_images = np.array(gauss_images)
        # 8 x 2 x 128 x 128: sigma(50000) x surface(2 different models) x 128 x 128
        print('gauss images shape: ', gauss_images.shape)
        gauss_filtered.append(gauss_images)
        
        # For pixelwise plot: 128 x 128 (averaged over different products for the same region e.g. 0, 488)
        gauss_rmsd = compute_avg_rmsd(gauss_images, target, hw_size=5, mask=mask)
        print('GAUSS RMSD SHAPE: ', gauss_rmsd.shape, gauss_rmsd)
        # Number of sigmas x 128 x 128
        gauss_rmsds.append(gauss_rmsd)

        # For line graph: 2
        gauss_avg_rmsds.append(np.mean(gauss_rmsd))
    print('GAUSS RMSDS SHAPE: ', len(gauss_rmsds), gauss_rmsds)
    # Following means pixelwise plots and gives out 1 value per sigma value
    print('GAUSS AVG RMSDS SHAPE: ', len(gauss_avg_rmsds), gauss_avg_rmsds)



    # Calculate RMSD: test for one filter radius
    # -----------------------
    # Avg RMSD for one sigma:
    gauss_rmsd = compute_avg_rmsd(gauss_filtered[0], target, hw_size=5, mask=mask)
    # plt.imshow(gauss_rmsd, cmap='jet', vmax=0.2)
    # plt.show()

    def forward_pass(model, network_images, target=None):
        outputs = model(network_images)
        n_images, outputs = detach_tensors([network_images, outputs])
        n_mask = land_false(n_images)[0]
        if target is not None:  # I don't know why this is here
            target = target * mask
        n_images = n_images * n_mask
        outputs = outputs * n_mask
        return outputs


    def evaluate_model(model, network_images):
        outputs = forward_pass(model, network_images)
        rmsd = compute_avg_rmsd(outputs[:, 0], target, hw_size=5, mask=mask)
        avg_residual = compute_avg_residual(outputs, target)
        residual = (outputs[0] - target[0])

        return outputs, rmsd, avg_residual, residual[0]


    def test_multiple_models(model_filepaths):
        """Runs multiple models and returns lists of metrics

        Args:
            model_filepaths (list of str): Filepaths to the models to be tested.
        """
        rmsds = []
        avg_residuals = []
        network_avg_rmsds = []
        network_outputs = []
        residuals = []
        for filepath in model_filepaths:
            model.load_state_dict(torch.load(filepath))
            outputs, rmsd, avg_residual, residual = evaluate_model(model, network_images)

            network_outputs.append(outputs)
            rmsds.append(rmsd)
            network_avg_rmsds.append(np.mean(rmsd))
            avg_residuals.append(avg_residual)
            residuals.append(residual)

        return rmsds, avg_residuals, network_avg_rmsds, network_outputs, residuals
    

    # Computing RMSD for region (x,y) across different number of epochs
    # --------------------------------------------------
    r = 2
    a = 5
    epoch_inc = 25
    epochs = [a * r ** i for i in range(6)]
    vanilla_filepaths = [f'./models/vanilla_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    rmsds, avg_residuals, network_avg_rmsds, network_outputs, residuals = test_multiple_models(vanilla_filepaths)

    aug_filepaths = [f'./models/aug_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    aug_rmsds, aug_avg_residuals, aug_network_avg_rmsds, aug_network_outputs, aug_residuals = test_multiple_models(aug_filepaths)

    multiscale_loss_filepaths = [f'./models/multiscale_loss_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    ms_rmsds, ms_avg_residuals, ms_network_avg_rmsds, ms_network_outputs, ms_residuals = test_multiple_models(multiscale_loss_filepaths)

    aug_multiscale_loss_filepaths = [f'./models/aug_multiscale_loss_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    aug_ms_rmsds, aug_ms_avg_residuals, aug_ms_network_avg_rmsds, aug_ms_network_outputs, aug_ms_residuals = test_multiple_models(aug_multiscale_loss_filepaths)

    WAE_loss_filepaths = [f'./models/WAE_vanilla_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    WAE_rmsds, WAE_avg_residuals, WAE_network_avg_rmsds, WAE_network_outputs, WAE_residuals = test_multiple_models(WAE_loss_filepaths)

    WAE_coast_filepaths = [f'./models/coast_WAE_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    WAE_coast_rmsds, WAE_coast_avg_residuals, WAE_coast_network_avg_rmsds, WAE_coast_network_outputs, WAE_coast_residuals = test_multiple_models(WAE_coast_filepaths)

    # Providing first images for following subplots (Gaussian filtered for RMSD and Unfiltered for Residuals)
    # -----------------------------------------------
    min_index = gauss_avg_rmsds.index(min(gauss_avg_rmsds))
    rmsds = [gauss_rmsds[min_index]] + rmsds
    avg_residuals = [compute_avg_residual(images, target)] + avg_residuals
    residuals = [(images[0] - target[0])] + residuals

    #
    # -----------------------------------------------
    indices = [0,3,6,9,12,14]
    plot_list = [network_outputs[i][0][0] for i in range(6)] + [WAE_network_outputs[i][0][0] for i in range(6)] + [WAE_coast_network_outputs[i][0][0] for i in range(6)] + [gauss_filtered[i][0] for i in indices]
    plot_titles = [f'{epoch} epochs' for epoch in epochs]*3 + [f'filter radius: {sigmas[i]//1000}km' for i in indices]
    create_subplot(plot_list, [[x, y]] * len(plot_list), cols=6, titles=plot_titles)
    plt.show()


    #
    # -----------------------------------------------
    indices = [0, 2, 5]
    plot_list = [images[0]] + [network_outputs[i][0][0] for i in indices] + [target[0]]
    create_subplot(plot_list, [[x, y]] * len(plot_list), cols=5)
    plt.show()
    plot_list = [images[0]-target[0]] + [network_outputs[i][0][0]-target[0] for i in indices]
    create_subplot(plot_list, [[x, y]] * len(plot_list), residual=True, cols=4)
    plt.show()


    # create_subplot(rmsds, [[x, y]] * batch_size, mdt=mdt, rmse=True, titles=['Gaussian Filtered'] + [f'{epoch} epochs' for epoch in epochs], big_title=f'RMSD between network output and reference - trained over a different number of epochs ({var} trained)')    
    # create_subplot(residuals, [[x, y]] * batch_size, mdt=mdt, rmse=False, residual=True, titles=['Unfiltered'] + [f'{epoch} epochs' for epoch in epochs], big_title=f'Residual difference (Output - Reference) trained over a different number of epochs ({var} trained)')
    
    # Plot line graph of epochs against Gaussian RMSD
    # ----------------------------------------------
    print('gauss rmses :', gauss_avg_rmsds)
    print('network rmses :', network_avg_rmsds)
    plt.plot(sigmas, gauss_avg_rmsds)
    plt.axhline(y=np.min(network_avg_rmsds), color='r', linestyle='dashed')
    plt.legend(["Gaussian filter output", "Best CDAE Output"], loc ="upper right")
    plt.xlabel('Gaussian Filter Radius', fontsize=12)
    # plt.xlabel('Gaussian Filter Half-weight Radius (km)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    # plt.ylim(bottom=0.14, top=0.23)
    plt.title('RMSE of Different Filtering Methods Against ' + ref_var + ' Data')
    plt.show()
    plt.close()

    epochs = np.arange(10, 190, 10)
    vanilla_model_filepaths = [f'./models/vanilla_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    _, _, vanilla_avg_rmsds, _, _ = test_multiple_models(vanilla_model_filepaths)
    plt.plot(epochs, vanilla_avg_rmsds)

    # Generate avg_rmsds for other networks
    aug_model_filepaths = [f'./models/aug_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    _, _, aug_avg_rmsds, _, _ = test_multiple_models(aug_model_filepaths)
    plt.plot(epochs, aug_avg_rmsds)

    ms_model_filepaths = [f'./models/multiscale_loss_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    _, _, ms_avg_rmsds, _, _ = test_multiple_models(ms_model_filepaths)
    plt.plot(epochs, ms_avg_rmsds)

    aug_multi_filepaths = [f'./models/aug_multiscale_loss_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    _, _, aug_multi_avg_rmsds, _, _ = test_multiple_models(aug_multi_filepaths)
    plt.plot(epochs, aug_multi_avg_rmsds)

    plt.axhline(y=np.min(gauss_avg_rmsds), color='r', linestyle='dashed')
    plt.legend(['Vanilla', 'Augmentation', 'Multiscale Loss', 'Multiscale Loss + Augmentation', 'Gaussian filter'], loc ="upper right")
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    # plt.ylim(bottom=0.14, top=0.23)
    plt.title('RMSE of Different Filtering Methods Against ' + ref_var + ' Data')
    plt.show()
    plt.close()

    vanilla_model_filepaths = [f'./models/vanilla_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    _, _, vanilla_avg_rmsds, _, _ = test_multiple_models(vanilla_model_filepaths)
    plt.plot(epochs, vanilla_avg_rmsds)

    WAE_model_filepaths = [f'./models/WAE_vanilla_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    _, _, WAE_avg_rmsds, _, _ = test_multiple_models(WAE_model_filepaths)
    plt.plot(epochs, WAE_avg_rmsds)

    WAE_coast_filepaths = [f'./models/coast_WAE_{var}/{epoch}e_{var}_model_cdae.pth' for epoch in epochs]
    _, _, WAE_coast_avg_rmsds, _, _ = test_multiple_models(WAE_coast_filepaths)
    plt.plot(epochs, WAE_coast_avg_rmsds)

    plt.axhline(y=np.min(gauss_avg_rmsds), color='r', linestyle='dashed')
    plt.legend(['GAN Noise', 'WAE Noise', 'WAE Noise trained on coastal regions', 'Gaussian filter'], loc ="upper right")
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('RMSE of Different Filtering Methods Against ' + ref_var + ' Data')
    plt.show()
    plt.close()



    rmses = []
    for x, y in zip(x_coords, y_coords):
        regions = dataset.get_regions(x, y)
        target = t_dataset.get_regions(x, y)[0]
        regions = torch.stack(regions)
        output_regions = model(regions)
        target = target.detach().cpu().numpy()
        output_regions = output_regions.detach().cpu().numpy()
        mask = target != 0

        rmse = np.sqrt(np.mean((output_regions - target)**2, axis=0))
        rmse = rmse * mask
        rmses.append(rmse[0])



if __name__ == "__main__":
    main()



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




