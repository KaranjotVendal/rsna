import time
import numpy as np
import pandas as pd

import torch
import torchmetrics

from config import config
import wandb


def evaluate(model, test_loader, fold, mod, device):
    test_time = time.time()

    checkpoint = torch.load(f'./checkpoints/{config.MODEL}_model_{mod}_{fold}.pth')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
    test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average='macro')       
    test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2)

    test_targets = []
    preds = []    
    for e, batch in enumerate(test_loader):
        with torch.no_grad():
    
            features = batch['X'].to(device)
            targets = batch['y'].to(device)
            org = batch['org']
            
            logits, probs = model(features, org)
            test_targets.append(targets)
            preds.append(probs)
    
    test_targets = torch.cat(test_targets).flatten()
    preds = torch.cat(preds)

    acc = test_acc(preds.cpu(), test_targets.cpu())
    f1 = test_f1(preds.cpu(), test_targets.cpu())
    auroc = test_auroc(preds.cpu(), test_targets.cpu())

    if config.WANDB:
        wandb.log({
                'test acc': acc,
                'test f1': f1,
                'test AUROC': auroc
                })

    elapsed_time = time.time() - test_time
    
    print(f'\nInference complete for {mod} in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s | Accuracy: {acc:.3f}% | F1 Score: {f1:.4f} | AUROC: {auroc:.4f}')


    return acc.item(), f1.item(), auroc.item() 