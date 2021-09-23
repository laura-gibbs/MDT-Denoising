import glob
import os
import subprocess
import shutil
import numpy as np

filenames = glob.glob('../../a_mdt_data/computations/mdts/mdts_tbf/*.dat')
filenames = [os.path.split(fname)[-1] for fname in filenames]
filenames = [fname[:len(fname)-4] for fname in filenames]

sigmas = np.arange(30000, 151000, 10000)

for fname in filenames:
    for sigma in sigmas:
        with open('filter_params.txt', 'w+') as f:
            f.write(fname)
            f.write('\n4')
            f.write('\n' + str(sigma))
        subprocess.run('./spatial_wmn_filter')
        # shutil.move("../a_mdt_data/computations/currents/" + fname + '_cs.dat', "../a_mdt_data/computations/currents/gauss_filt_geod_mdts_cs/" + fname + '_cs.dat')
