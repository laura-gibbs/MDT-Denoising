import glob
import os
from skimage import io, transform
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import random

class CAEDataset(Dataset):

    def __init__(self, region_dir, quilt_dir, mdt=False, transform=None):
            self.region_dir = region_dir
            self.quilt_dir = quilt_dir
            self.testing = quilt_dir is None
            self.transform = transform
            self.mdt = mdt
            self.paths = glob.glob(os.path.join(region_dir, '*.npy'))
            if self.quilt_dir is not None:
                self.quilt_paths = glob.glob(os.path.join(quilt_dir, '*.png'))
            print(region_dir)


    def __len__(self):
        return len(self.paths)

    
    def get_regions(self, x, y):
        indices = []
        for i in range(len(self.paths)):
            split_path = self.paths[i][:len(self.paths[i])-4].split('_')
            a, b = int(split_path[-2]), int(split_path[-1])
            if x == a and y == b:
                indices.append(i)
        regions = []
        for i in indices:
            region, _ = self[i]
            regions.append(region)
        return regions


    def __getitem__(self, idx):
        input_name = self.paths[idx]
        target_img = np.load(input_name)
        
        if self.testing:
            return ToTensor()(target_img), None   
        
        # Load and apply random quilt
        random_quilt = random.choice(self.quilt_paths)
        quilt = Image.open(random_quilt)
        quilt = quilt.convert(mode='L')
        quilt = np.array(quilt).astype(np.float32)
        mask = target_img != 0
        quilt = (quilt - np.nanmin(quilt)) / (np.nanmax(quilt) - np.nanmin(quilt))
        if self.mdt:
            quilt = (quilt*2) - 1
        img = target_img + .3* quilt * mask

        # consider turning back to PIL images for transforms e.g. rotations, flips
        # img = Image.fromarray(img)
        # target_img = Image.fromarray(target_img)

        if self.transform is not None:
            img = self.transform(img)
            target_img = self.transform(target_img)


        return ToTensor()(img), ToTensor()(target_img)