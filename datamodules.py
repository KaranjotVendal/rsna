import os
import glob

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from utils import load_image




# [batch, max_slices, ch, H, W], orginal no of slices before padding = []
class RSNAdataset(Dataset):
    def __init__(self, patient_path, paths, targets, n_slices, img_size, transform=None):
        #(self, './data/reduced_dataset/', t['xtrain'],t['ytrain'], 254, 112, transform)
        self.patient_path = patient_path
        self.paths = paths
        self.targets = targets
        self.n_slices = n_slices
        self.img_size = img_size
        self.transform = transform
          
    def __len__(self):
        return len(self.paths)
    
    def padding(self, paths): 
        images=[load_image(path) for path in paths]
        org_size = len(images)
            
        dup_len = 254 - len(images)
        if org_size == 0:
            dup = torch.zeros(1, 112, 112)
        else:
            dup = images[-1]
        for i in range(dup_len):
            images.append(dup)

        images = [torch.tensor(image, dtype=torch.float32) for image in images]
        images = torch.stack(images)

        return images, org_size
    
    def __getitem__(self, index):
        _id = self.paths[index]
        patient_path = os.path.join(self.patient_path, f'{str(_id).zfill(5)}/')

        data = []
        org = []
        for t in ["FLAIR", "T1w", "T1wCE", "T2w"]:
            t_paths = sorted(
                glob.glob(os.path.join(patient_path, t, "*")), 
                key=lambda x: int(x[:-4].split("-")[-1]),
            )
            num_samples = self.n_slices
            image, org_size = self.padding(t_paths)

            data.append(image)
            org.append(org_size)
            break
            
        data = torch.stack(data).transpose(0,1)
        y = torch.tensor(self.targets[index])
        
        return {"X": data.float(), "y": y, 'org': org}


































