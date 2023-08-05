import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import lightning as L

from utils import load_image

import os
import glob

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
            dup = torch.zeros(self.n_slices, 112, 112)
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
        y = torch.tensor(self.targets[index], dtype=torch.float32)
        
        return {"X": data.float(), "y": y}, org




















































class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_path="./", batch_size=64, num_workers=0, height_width=(28,28)):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.height_width = height_width
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(root=self.data_path, download=True)
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(self.height_width),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(self.height_width),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ]
        )
        return

    def setup(self, stage=None):
        # Note transforms.ToTensor() scales input images
        # to 0-1 range
        train = datasets.MNIST(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=False,
        )

        self.test = datasets.MNIST(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=False,
        )

        self.train, self.valid = random_split(train, lengths=[55000, 5000])

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader