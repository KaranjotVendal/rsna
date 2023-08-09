import os
import time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
#import albumentations as A
#import argparse

from models import RACNet
from trainer import Trainer
from datamodules import RSNAdataset
from utils import LossMeter

import wandb


def train(MODEL, lr, num_classes, path, epochs, n_fold, batch_size, num_workers, device):
    
    fold_acc = []
    fold_auroc = []
    fold_f1 = []
    
    start_time = time.time()
    model = RACNet(MODEL, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy

    '''folds_xtrain = np.load('./data/folds/new_folds/xtrain.npy', allow_pickle=True)
    folds_xtest = np.load('./data/folds/new_folds/xtest.npy', allow_pickle=True)
    folds_ytrain = np.load('./data/folds/new_folds/ytrain.npy', allow_pickle=True)
    folds_ytest = np.load('./data/folds/new_folds/ytest.npy', allow_pickle=True)'''

    dlt = []
    empty_fld = [109, 123, 709]
    df = pd.read_csv("data/train_labels.csv")
    skf = StratifiedKFold(n_splits=10)
    X = df['BraTS21ID'].values
    Y = df['MGMT_value'].values

    for i in empty_fld:
        j = np.where(X == i)
        dlt = dlt.append(j)
        X = np.delete(X, j)
        
    Y = np.delete(Y,dlt)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):  
    
        print(f'--------------FOLD:{fold}-----------------------') 
        
        xtrain = X[train_idx]
        ytrain = Y[train_idx]
        xtest = X[test_idx]
        ytest = Y[test_idx]

        train_set = RSNAdataset(
                        './data/reduced_dataset/',
                        xtrain,  
                        ytrain,
                        n_slices=254,
                        img_size=112,
                        transform=None
                            )
    
        test_set = RSNAdataset(
                        './data/reduced_dataset/',
                        xtest,  
                        ytest,
                        n_slices=254,
                        img_size=112,
                        transform=None
                            )
        
        train_loader = DataLoader(
                    train_set,    
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                )
        
        test_loader = DataLoader(
                    test_set,    
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
        
        trainer = Trainer(
                    model, 
                    device, 
                    optimizer, 
                    criterion,
                    epochs,
                    LossMeter, 
                    fold
                    )
        
        trainer.fit(epochs,
                    train_loader,
                    save_path = './checkpoints/f"RACNet_model-{fold}.pth',
                    patience = 5
                   )
                        
        #testing loop
        test_acc, test_f1, test_auroc = trainer.test(test_loader)

        fold_acc.append(test_acc)
        fold_f1.append(test_f1)
        fold_auroc.append(test_auroc)
        
        wandb.log({
         'FOLD f1 score':avg_acc,
         'FOLD f1 score':avg_f1,
         'FOLD AUROC': avg_auroc,
         })
    
    f1_std = torch.cat(fold_f1)
    std = torch.std(f1_std)
    
    elapsed_time = time.time() - start_time
    
    
    print('\nCross validation loop complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('fold accuracy:', fold_acc)
    print('fold f1_score:',fold_f1)
    print('fold auroc:', fold_auroc)
    print('Std F1 score:'.format(std))
    
    
    wandb.finish()






