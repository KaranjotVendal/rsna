import os
import glob
import random
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset

from utils import load_image
from config import config



# [batch, max_slices, C, H, W], orginal no of slices before padding = []
class RSNAdataset(Dataset):
    def __init__(self, patient_path, paths, targets, n_slices, img_size, type='FLAIR', transform=None):
       
        self.patient_path = patient_path
        self.paths = paths
        self.targets = targets
        self.n_slices = n_slices
        self.img_size = img_size
        self.transform = transform
        self.type = type
          
    def __len__(self):
        return len(self.paths)
    
    def padding(self, paths): 
        images=[load_image(path) for path in paths]
        org_size = len(images)
        
        if config.ALBUMENTATION:
            if self.transform:
                seed = random.randint(0,99999)
                for i in range(len(images)):
                    random.seed(seed)
                    images[i] = self.transform(image=images[i])["image"]

                images = [torch.tensor(image, dtype=torch.float32) for image in images]
        
        elif config.PYTORCH_TRANSFORM:
            images = [torch.tensor(image, dtype=torch.float32) for image in images]
            if self.transform:
                seed = random.randint(0,99999)
                for i in range(len(images)):
                    random.seed(seed)
                    images[i] = self.transform(images[i])
       
        dup_len = self.n_slices - org_size
        if org_size == 0:
            dup = np.zeros((1, self.img_size, self.img_size))
        else:
            dup = images[-1]
        for i in range(dup_len):
            images.append(dup)

        if self.transform is None:
            images = [torch.tensor(image, dtype=torch.float32) for image in images]
        images = torch.stack(images)     #[n_slices, C, H, W]
       
        return images, org_size
    
    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = os.path.join(self.patient_path, f'{str(_id).zfill(5)}/')

        data = []
        org = []
        t_paths = sorted(
            glob.glob(os.path.join(patient_path, self.type, "*")), 
            key=lambda x: int(x[:-4].split("-")[-1]),
        )
        
        image, org_size = self.padding(t_paths)

        data.append(image)
        org.append(org_size)
        
        if self.transform:
            if config.PYTORCH_TRANSFORM:
                data = torch.stack(data).squeeze(0) #.transpose(0,1)
            else:
                data = torch.stack(data).transpose(0,1)
        else:
            data = torch.stack(data).transpose(0,1)
        
        y = torch.tensor(self.targets[index])
        
        return {"X": data.float(), "y": y, 'org': org}