
from torch.utils.data import Dataset
import torch
import numpy as np
import os
import re
from skimage import exposure
from copy import deepcopy
from tqdm import tqdm

class cellsDataset(Dataset):
    def __init__(self, images_path, somas_path, balance_data = False, transform=None, label=True, hist_equal=False, load2Ram=False):
        self.images_path = images_path
        self.somas_path = somas_path
        self.transform = transform
        self.balance_data = balance_data
        self.label = label
        self.hist_equal = hist_equal
        self.load2Ram = load2Ram
        self.make_paths()
        if self.load2Ram:
            self.load_all()
        

    def make_paths(self):
        self.images = os.listdir(self.images_path)
        if self.label:
            self.somas = os.listdir(self.somas_path)
        self.images.sort(key=lambda f: int(re.sub('\D', '', f)))
        if self.label:
            self.somas.sort(key=lambda f: int(re.sub('\D', '', f)))
        self.images = [os.path.join(self.images_path, img) for img in self.images]
        if self.label:
            self.somas = [os.path.join(self.somas_path, soma) for soma in self.somas]
            

    def __len__(self):
        return len(self.images)
    

    def load_all(self):
        self.samples = []
        print('Loading the data into the RAM')
        for idx in tqdm(range(len(self.images))):
            image_paths = self.images[idx]
            if self.label:
                soma_paths = self.somas[idx]

            image = np.load(image_paths)
            if self.label:
                soma = np.load(soma_paths)

            if self.hist_equal:
                img = deepcopy(image['arr_0'])
                img = exposure.equalize_adapthist(np.squeeze(np.array(image['arr_0'])),5)*255
            else:
                img = image['arr_0']

            if self.label:
                sample = {'image': img, 'soma': soma['arr_0']}
            else:
                sample = {'image': image['arr_0']}
            
            sample['image'] = np.expand_dims(sample['image'], 0) # Batch Size = 1
            if self.label:
                sample['soma'] = np.expand_dims(sample['soma'], 0)

            sample['image'] = torch.tensor(sample['image'], dtype=torch.float32)
            if self.label:
                sample['soma'] = torch.tensor(sample['soma'], dtype=torch.bool)
            

            self.samples.append(sample) 


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if self.load2Ram:
            sample = self.samples[idx]
        else:
            image_paths = self.images[idx]
            if self.label:
                soma_paths = self.somas[idx]

            image = np.load(image_paths)
            if self.label:
                soma = np.load(soma_paths)

            if self.hist_equal:
                img = deepcopy(image['arr_0'])
                img = exposure.equalize_adapthist(np.squeeze(np.array(image['arr_0'])),5)*255
            else:
                img = image['arr_0']

            if self.label:
                sample = {'image': img, 'soma': soma['arr_0']}
            else:
                sample = {'image': image['arr_0']}
            
            sample['image'] = np.expand_dims(sample['image'], 0) # Batch Size = 1
            if self.label:
                sample['soma'] = np.expand_dims(sample['soma'], 0)

            sample['image'] = torch.tensor(sample['image'], dtype=torch.float32)
            if self.label:
                sample['soma'] = torch.tensor(sample['soma'], dtype=torch.bool)


        if self.transform and self.label:
            sample = self.transform(sample)

        return sample