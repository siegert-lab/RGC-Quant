# import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import mat73
import numpy as np
import json
import torchio as tio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from copy import deepcopy, copy
import tifffile as tiff
from scipy import ndimage
import click
from math import floor
from scipy.ndimage import grey_dilation

from models import CellSomaSegmentationModel

from utils import *
from datasets import cellsDataset

@click.group(chain=True)
def cli():
    pass

@cli.command('RGC_detection')
@click.option('--file_path', '-fp', type=str, required=True, help='Path to the image file or directory containing image files (.ims files supported for now)')
def fluer_pipeline(file_path):
    if os.path.isfile(file_path):
        file_paths = [file_path]
    else:
        file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.endswith('.ims')]
    
    for file_path in file_paths:
        print(f"file_path: {file_path}")

        img_path = file_path
        dataset_name = os.path.basename(img_path).split(".")[0]
        print(f"dataset_name: {dataset_name}")
        mat_path = os.path.join("tmp", dataset_name, 'mat')
        save_path_dataset = os.path.join("tmp", dataset_name, 'dataset')

        print(f"img_path: {img_path}")
        print("=======Make mat fules out of img/nd2 file=======")

        if not os.path.exists(save_path_dataset):
            os.makedirs(save_path_dataset)
        
        print(f"=======Make Masks Soma=======")
        datasets_path_train = save_path_dataset 
        num_epochs = 1000
        print(num_epochs)

        load2Ram = False
        hist_equal = False
        transform = None

        datasets_paths = [os.path.join(datasets_path_train, folder) for folder in os.listdir(datasets_path_train) if os.path.isdir(os.path.join(datasets_path_train, folder))]
        print(datasets_paths)
        datasets_list_train = []
        print("128")
        for dataset_path in tqdm(datasets_paths):
            img_datapath = os.path.join(dataset_path, 'img')
            soma_datapath = os.path.join(dataset_path, 'soma')
            metadata_path = os.path.join(dataset_path, 'metadata.json')
            # try:
            # load metadata (Not used)
            with open(metadata_path) as f:
                metadata = json.load(f)
            print(metadata)
            print("124")

            cells_dataset = cellsDataset(img_datapath, soma_datapath, transform=transform, hist_equal=hist_equal, load2Ram=load2Ram, label=False)
            
            datasets_list_train.append(cells_dataset)
            print(f'Loading dataset: {dataset_path} ==> Loaded')
            # except:
            #     print(f'Loading dataset: {dataset_path} ==> Failed')


        # concatenate all datasets
        dataset_train = torch.utils.data.ConcatDataset(datasets_list_train)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model = CellSomaSegmentationModel()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(os.path.join('./models/soma_detection.pth')))
        model.eval()
        model.to(device)

        len_dataset = floor(len(dataset_train))
        size_sample = np.squeeze(dataset_train.__getitem__(0)['image']).shape

        images = np.zeros((len_dataset, size_sample[0], size_sample[1], size_sample[2]), dtype=np.uint8)
        # soma = np.zeros((len_dataset, size_sample[0], size_sample[1], size_sample[2]), dtype=np.uint8)
        segmented_images = np.zeros((len_dataset, size_sample[0], size_sample[1], size_sample[2]), dtype=np.uint8)

        for i in tqdm(range(len_dataset)):
            sample = dataset_train.__getitem__(i)
            images[i, :, :, :] = np.array(sample['image'], dtype=np.uint8)
            # soma[i, :, :, :] = np.array(sample['soma'], dtype=np.uint8)

            segmented = model(torch.tensor(np.expand_dims(sample['image'], 0)).type(torch.float32).to(device))
            segmented = segmented.squeeze(0).cpu().detach().numpy()
            segmented = np.uint8(segmented*255)
            segmented_images[i, :, :, :] = segmented

        images =  reconstruct_image(images, metadata)
        segmented_images =  reconstruct_image(segmented_images, metadata)

        # Create a 3D structuring element
        structuring_element = np.ones((5, 5, 5))

        # Ensure your image is uint8
        segmented_images = segmented_images > 125
        segmented_images = segmented_images.astype(np.uint8)

        # Dilate the imageow
        dilated_image = grey_dilation(segmented_images, footprint=structuring_element)

        _, num_labels = ndimage.label(dilated_image)

        print(f"Number of cells: {num_labels+1}")
        model_save_path_soma = os.path.join("tmp", dataset_name, "soma_out")
        if not os.path.exists(model_save_path_soma):
            os.makedirs(model_save_path_soma, exist_ok=True)
        # Write the number of cells to a text file
        with open(os.path.join(model_save_path_soma,'number_of_cells.txt'), 'w') as file:
            file.write(str(num_labels+1))
        print("Save Starting")

        images = np.transpose(images, [2,0,1])
        segmented_images = np.transpose(segmented_images, [2,0,1])
        for img, name in zip([images,segmented_images], ['img','segmented']): 
            name = f'{name}.tif'
            path = os.path.join(model_save_path_soma, name)
            print(path)
            tiff.imwrite(path, img)

if __name__ == '__main__':
    cli()