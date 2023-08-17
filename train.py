import time
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold

import albumentations as A
#import argparse

from models import *
from trainer import Trainer
from datamodule import RSNAdataset
from utils import LossMeter
from eval import evaluate
from config import config

import wandb


def train():
    
    fold_acc = []
    fold_auroc = []
    fold_f1 = []

    
    start_time = time.time()
    model = Res18GRU(config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = F.cross_entropy

    dlt = []
    empty_fld = [109, 123, 709]
    df = pd.read_csv("data/train_labels.csv")
    skf = StratifiedKFold(n_splits=config.KFOLD)
    X = df['BraTS21ID'].values
    Y = df['MGMT_value'].values

    for i in empty_fld:
        j = np.where(X == i)
        dlt.append(j)
        X = np.delete(X, j)
        
    Y = np.delete(Y,dlt)

    '''train_transform = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(
                                    shift_limit=0.0625, 
                                    scale_limit=0.1, 
                                    rotate_limit=10, 
                                    p=0.5
                                ),
                                A.RandomBrightnessContrast(p=0.5),
                            ])'''

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):  
    
        print(f'--------------FOLD:{fold}-----------------------') 
        
        xtrain = X[train_idx]
        ytrain = Y[train_idx]
        xtest = X[test_idx]
        ytest = Y[test_idx]

        train_set = RSNAdataset(
                        config.DATA_PATH,
                        xtrain,  
                        ytrain,
                        n_slices=config.N_SLICES,
                        img_size=config.IMG_SIZE,
                        type = config.MOD,
                        transform=None
                            )
    
        test_set = RSNAdataset(
                        config.DATA_PATH,
                        xtest,  
                        ytest,
                        n_slices=config.N_SLICES,
                        img_size=config.IMG_SIZE,
                        type = config.MOD,
                        transform=None
                            )
        
        train_loader = DataLoader(
                    train_set,    
                    batch_size=config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=config.NUM_WORKERS,
                )
        
        test_loader = DataLoader(
                    test_set,    
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=config.NUM_WORKERS,
                )
        
        trainer = Trainer(
                    model, 
                    config.DEVICE, 
                    optimizer, 
                    criterion,
                    config.NUM_EPOCHS,
                    LossMeter, 
                    fold
                    )
        
        #test_acc, test_f1, test_auroc = 
        trainer.fit(train_loader,
                    test_loader,
                    save_path = f'./checkpoints/{config.MODEL}_model_{config.MOD}_{fold}.pth',
                    )

        acc, f1, auroc = evaluate(test_loader,
                                fold,
                                config.MOD,
                                config.DEVICE)
        
        fold_acc.append(acc)
        fold_f1.append(f1)
        fold_auroc.append(auroc)

    
    elapsed_time = time.time() - start_time
    
    
    print('\nCross validation loop complete for {} in {:.0f}m {:.0f}s'.format(config.MOD, elapsed_time // 60, elapsed_time % 60))
    print('\nfold accuracy:', fold_acc)
    print('\nfold f1_score:',fold_f1)
    print('\nfold auroc:', fold_auroc)
    print('\nStd F1 score:', np.std(np.array(fold_f1)))
    print('\nAVG performance of model:', np.mean(np.array(fold_f1)))
    
    wandb.finish()