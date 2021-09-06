import glob
import os

files = glob.glob('*.dat')
for file in files:
    os.rename(file, file.replace('.npy',''))
