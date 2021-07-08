import numpy as np
from PIL import Image
from trad_filters.gaussian_filter import rb_gaussian
import matplotlib.pyplot as plt
from utils.utils import get_spatial_params
from utils.read_data import read_surface


# mdt = read_surface('dtu18_gtim5_do0280_rr0004.dat', '../a_mdt_data/computations/mdts')
# plt.imshow(np.rot90(mdt))
# plt.show()
mdt = np.load('../a_mdt_data/geodetic_cdae/mdt/dtu18_egm2008_do0280_rr0004_0_488.npy')
# plt.imshow(cs)
# plt.show()
mask = mdt == 0 # should this be ==0?
# print('mask', mask.shape)
# plt.imshow(mask)
# plt.show()
lons, lats, _ = get_spatial_params(0, 488)
print(lons,lats)
II, JJ = 128, 128
# mask = np.zeros((II, JJ))
# mask[0] = 1
print(II, JJ)
# mdt = mdt.tranpose()

print(mdt.shape)
gauss_mdt, residual = rb_gaussian(II, JJ, lons, lats, mdt, mask, 50000, 'gsn')
fig, ax = plt.subplots(1, 2)
ax[0].imshow(mdt)
ax[1].imshow(gauss_mdt)
plt.show()
