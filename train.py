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
    fold_loss = []
    fold_auroc = []
    fold_f1 = []

    start_time = time.time()
    model = RACNet(MODEL, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = F.cross_entropy
        
    for _ in range(n_fold):
        fold = _+1
        folds_xtrain = np.load('./data/folds/new_folds/xtrain.npy', allow_pickle=True)
        folds_xtest = np.load('./data/folds/new_folds/xtest.npy', allow_pickle=True)
        folds_ytrain = np.load('./data/folds/new_folds/ytrain.npy', allow_pickle=True)
        folds_ytest = np.load('./data/folds/new_folds/ytest.npy', allow_pickle=True)
        
        xtrain = folds_xtrain[_]
        ytrain = folds_ytrain[_]
        xtest = folds_xtest[_]
        ytest = folds_ytest[_]

        print('-'*30)
        print(f"Fold {fold}")

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
                    shuffle=True,
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
                        
        #trainer.plot_loss()
        #trainer.plot_score()
        #trainer.plot_fscore()
                
        #testing loop
        loss, test_acc, test_f1, test_auroc = trainer.test(test_loader)
        fold_loss.append(loss)
        fold_acc.append(test_acc)
        fold_f1.append(test_f1)
        fold_auroc.append(test_auroc)    
    
    elapsed_time = time.time() - start_time
    '''wandb.log({
         'Avg Test f1 score': np.mean(test_fscore),
         'Avg Train f1 score': np.mean(f_scores)
         })'''
    
    
    print('\nCross validation loop complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Avg loss {:.5f}'.format(np.mean(losses)))
    print('Avg score {:.5f}'.format(np.mean(scores)))
    print('Avg Train f1_score {:.5f}'.format(np.mean(f_scores)))
    print('Avg Test f1_score {:.5f}'.format(np.mean(test_fscore)))






