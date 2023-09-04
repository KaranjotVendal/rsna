import time
import numpy as np
import pandas as pd
import json

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold

import albumentations as A

from models import *
from trainer import Trainer
from datamodule import RSNAdataset
from utils import LossMeter, save_metrics_to_json, update_metrics
from eval import evaluate
from config import config
from plotting import plot_train_valid_fold, plot_train_valid_all_fold, plot_test_metrics

import wandb

def train():
    
    fold_acc = []
    fold_auroc = []
    fold_f1 = []

    metrics = {}

    start_time = time.time()

    dlt = []
    empty_fld = [109, 123, 709]
    df = pd.read_csv("data/train_labels.csv")
    skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=123)
    X = df['BraTS21ID'].values
    Y = df['MGMT_value'].values

    for i in empty_fld:
        j = np.where(X == i)
        dlt.append(j)
        X = np.delete(X, j)
        
    Y = np.delete(Y,dlt)

    train_transform = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(
                                    shift_limit=0.0625, 
                                    scale_limit=0.1, 
                                    rotate_limit=10, 
                                    p=0.5
                                ),
                            ])

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):  
    
        print(f'--------------FOLD:{fold}-----------------------') 
        metrics[fold] = {
        'train': {'loss': [], 'acc': [], 'f1': [], 'auroc': []},
        'valid': {'loss': [], 'acc': [], 'f1': [], 'auroc': []},
        'test': {'acc': [], 'f1': [], 'auroc': []}
        }
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
        
        if config.MODEL == 'Res18GRU':
            model = Res18GRU(config.NUM_CLASSES)
        if config.MODEL == 'Res18LSTM':
            model = Res18LSTM(config.NUM_CLASSES)
        if config.MODEL == 'Res50GRU':
            model = Res50GRU(config.NUM_CLASSES)
        if config.MODEL == 'Res50LSTM':
            model = Res50LSTM(config.NUM_CLASSES)
        if config.MODEL == 'ConvxGRU':
            model = ConvxGRU(config.NUM_CLASSES)
        if config.MODEL == 'ConvxLSTM':
            model = ConvxLSTM(config.NUM_CLASSES)    
        
        model = model.to(config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = F.cross_entropy
        trainer = Trainer(
                    model, 
                    config.DEVICE, 
                    optimizer, 
                    criterion,
                    config.NUM_EPOCHS,
                    LossMeter, 
                    fold
                    )
        
 
        trainer.fit(train_loader,
                    test_loader,
                    save_path = f'./checkpoints/{config.MODEL}_model_{config.MOD}_{fold}.pth',
                    )

        acc, f1, auroc = evaluate(model,
                                test_loader,
                                fold,
                                config.MOD,
                                config.DEVICE)

        for value in trainer.hist['train_loss']:
            update_metrics(metrics, fold, 'train', 'loss', value)
    
        for value in trainer.hist['train_acc']:
            update_metrics(metrics, fold, 'train', 'acc', value)
    
        for value in trainer.hist['train_f1']:
            update_metrics(metrics, fold, 'train', 'f1', value)
    
        for value in trainer.hist['train_auroc']:
            update_metrics(metrics, fold, 'train', 'auroc', value)
    
        for value in trainer.hist['val_loss']:
            update_metrics(metrics, fold, 'valid', 'loss', value)
    
        for value in trainer.hist['val_acc']:
            update_metrics(metrics, fold, 'valid', 'acc', value)
    
        for value in trainer.hist['val_f1']:
            update_metrics(metrics, fold, 'valid', 'f1', value)
    
        for value in trainer.hist['val_auroc']:
            update_metrics(metrics, fold, 'valid', 'auroc', value)

        update_metrics(metrics, fold, 'test', 'acc', acc)
        update_metrics(metrics, fold, 'test', 'f1', f1)
        update_metrics(metrics, fold, 'test', 'auroc', auroc)
        
        fold_acc.append(acc)
        fold_f1.append(f1)
        fold_auroc.append(auroc)

    json_path = save_metrics_to_json(metrics, config.MODEL)
    
    #plotting loss
    plot_train_valid_fold(json_path, 'loss')
    plot_train_valid_all_fold(json_path, 'loss')
    
    #plotting acc
    plot_train_valid_fold(json_path, 'acc')
    plot_train_valid_all_fold(json_path, 'acc')
    plot_test_metrics(json_path, 'acc')


    #plotting f1
    plot_train_valid_fold(json_path, 'f1')
    plot_train_valid_all_fold(json_path, 'f1')
    plot_test_metrics(json_path, 'f1')

    #plotting auroc
    plot_train_valid_fold(json_path, 'auroc')
    plot_train_valid_all_fold(json_path, 'auroc')
    plot_test_metrics(json_path, 'auroc')
   
    elapsed_time = time.time() - start_time
    
    
    print('\nCross validation loop complete for {} in {:.0f}m {:.0f}s'.format(config.MOD, elapsed_time // 60, elapsed_time % 60))
    print('\nfold accuracy:', fold_acc)
    print('\nfold f1_score:',fold_f1)
    print('\nfold auroc:', fold_auroc)
    print('\nStd F1 score:', np.std(np.array(fold_f1)))
    print('\nAVG performance of model:', np.mean(np.array(fold_f1)))

    if config.WANDB:
        wandb.log({
        'Avg performance f1': np.mean(np.array(fold_f1)),
        'Std f1 score': np.std(np.array(fold_f1)),
        'Avg performance acc': np.mean(np.array(fold_acc)),
        'Std acc score': np.std(np.array(fold_acc)),
        'Avg performance auroc': np.mean(np.array(fold_auroc)),
        'Std auroc score': np.std(np.array(fold_auroc)),
        })
    
        wandb.finish()