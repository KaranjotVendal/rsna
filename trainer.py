import time
import numpy as np

import torch
import torchmetrics
import wandb
from config import config

class Trainer():
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion,
        epochs,
        loss_meter, 
        fold
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.loss_meter = loss_meter
        self.fold = fold
        self.hist = {'val_loss':[],
                     'val_acc':[],
                     'val_f1':[],
                     'val_auroc':[],
                     'train_loss':[],
                     'train_acc':[],
                     'train_f1': [],
                     'train_auroc': [],
                    }
        
        self.best_test_auroc = -np.inf
        

    def fit(self, train_loader, test_loader, save_path, patience = 0):
        train_time = time.time()
        
        for epoch in range(self.epochs):
            t = time.time()
            self.model.train()
            train_loss = self.loss_meter()
            train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(self.device)
            train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average='macro').to(self.device)
            train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2).to(self.device)
            
            for idx, batch in enumerate(train_loader):
                features = batch['X'].to(self.device)
                targets = batch['y'].to(self.device)
                org = batch['org']
                logits, probs = self.model(features, org)
                loss = self.criterion(logits, targets)
                
                train_loss.update(loss.detach().item())
                train_acc.update(probs.detach(), targets)
                train_f1.update(probs.detach(), targets)
                train_auroc.update(probs.detach(), targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  
                
            _loss = train_loss.avg
            _acc = train_acc.compute()
            _f1 = train_f1.compute()
            _roc = train_auroc.compute()

            if config.WANDB:
                wandb.log({'train loss': _loss,
                        'train acc': _acc,
                        'train f1_score': _f1,
                        'train AUROC': _roc
                        })

            train_acc.reset()
            train_f1.reset()
            train_auroc.reset()
            
            self.hist['train_loss'].append(_loss)
            self.hist['train_acc'].append(_acc)
            self.hist['train_f1'].append(_f1)
            self.hist['train_auroc'].append(_roc)
            
            print(f' Train Epoch: {epoch+1}/{self.epochs} | Loss: {_loss:.5f} | Accuracy: {_acc:.4f}% | F1 Score: {_f1:.4f} | AUROC: {_roc:.4f} | Time: {time.time() - t}')
            
            val_loss, val_acc, val_f1, val_auroc = self.validate(test_loader, save_path)
            
            self.hist['val_loss'].append(val_loss)
            self.hist['val_acc'].append(val_acc)
            self.hist['val_f1'].append(val_f1)
            self.hist['val_auroc'].append(val_auroc)
            
        
        avg_loss = torch.mean(torch.tensor(self.hist['train_loss']))
        avg_acc = torch.mean(torch.tensor(self.hist['train_acc']))
        avg_f1 = torch.mean(torch.tensor(self.hist['train_f1']))
        avg_auroc = torch.mean(torch.tensor(self.hist['train_auroc']))

        print(f'Training Time: {(time.time() - train_time) // 60:.0f}m {(time.time() - train_time) % 60:.0f}s | Avg Loss: {avg_loss:.5f} | Avg Accuracy: {avg_acc:.3f}% | Avg F1 Score: {avg_f1:.4f} | Avg AUROC:{avg_auroc:.4f}')
        
            
    
    def validate(self, test_loader, save_path):
        test_time = time.time()
        self.model.eval()

        val_loss = self.loss_meter()    
        val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(self.device)
        val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average='macro').to(self.device)       
        val_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2).to(self.device)

        test_pred = []
        test_targets = []
        preds = []        
        for idx, batch in enumerate(test_loader):
            with torch.no_grad():
                features = batch['X'].to(self.device)
                targets = batch['y'].to(self.device)
                org = batch['org']

                logits, probs = self.model(features, org)
                loss = self.criterion(logits, targets)
                val_loss.update(loss.detach().item())
                
                test_targets.append(targets)
                preds.append(probs.detach())
                
                
        test_targets = torch.cat(test_targets).flatten()
        preds = torch.cat(preds)

        
        loss = val_loss.avg
        acc = val_acc(preds, test_targets)
        f1 = val_f1(preds, test_targets)
        auroc = val_auroc(preds, test_targets)             
                
        if auroc > self.best_test_auroc: 
            self.best_test_auroc = auroc

            torch.save({"model_state_dict": self.model.state_dict(),
                        "best_auroc": self.best_test_auroc,
                        },
                        save_path)
                      
            print(f'Checkpoint saved at {save_path} '
                  f'| Test acc: {acc :.2f}% '
                  f'| Test F1: {f1 :.3f}% '
                  f'| Best AUROC: {self.best_test_auroc:.3f}')           
            
        val_acc.reset()
        val_f1.reset()
        val_auroc.reset()
            
        if config.WANDB:
            wandb.log({'val loss':loss,
                    'val acc': acc,
                    'val f1_score': f1,
                    'val AUROC': auroc
                    })

        print(f"Validation Epoch: {(time.time() - test_time) // 60:.0f}m {(time.time() - test_time) % 60:.0f}s | Accuracy: {acc:.2f}% | F1 Score: {f1:.4f} | AUROC: {auroc:.4f}")
        return loss, acc.item(), f1.item(), auroc.item()