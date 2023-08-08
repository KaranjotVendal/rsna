import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchmetrics
from tqdm import tqdm
import wandb


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
        self.loss_meter = loss_meter
        self.fold = fold
        self.hist = {'test_loss':[],
                     'test_acc':[],
                     'test_f1':[],
                     'test_auroc':[],
                     'train_loss':[],
                     'train_acc':[],
                     'train_f1': [],
                     'train_auroc': [],
                    }
        
        #self.best_valid_score = -np.inf
        #self.best_valid_loss = np.inf
        #self.best_f_score = 0
        #self.n_patience = 0

        '''self.record = {'test_loss':[],
                     'test_acc':[],
                     'test_f1':[],
                     'test_auroc':[],
                     'train_loss':[],
                     'train_acc':[],
                     'train_f1': [],
                     'train_auroc': [],
                    }'''
        
        
    def fit(self, epochs, train_loader, save_path, patience):
        train_time = time.time()
        
        for epoch in range(epochs):
            t = time.time()
            self.model.train()
            train_loss = self.loss_meter()
            train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(self.device)
            train_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average='macro').to(self.device)
            train_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2).to(self.device)
            
            for idx, batch in enumerate(train_loader):
                print('-'*50)
                features = batch['X'].to(self.device)
                targets = batch['y'].to(self.device)
                org = batch['org']
                print(org)
                print("feature shape", features.shape)
                ### FORWARD AND BACK PROP
                logits, probs = self.model(features, org)
                loss = self.criterion(logits, targets)
                
                train_loss.update(loss.detach().item())
                train_acc.update(probs.detach(), targets)
                train_f1.update(probs.detach(), targets)
                train_auroc.update(probs.detach(), targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  
                print(f'-----------------Loss: {loss}-----------------------')
                print('------BATCH ENDING-------')
             
            _loss = train_loss.avg
            _acc = train_acc.compute()
            _f1 = train_f1.compute()
            _roc = train_auroc.compute()

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
            
            print(f'Epoch: {epoch+1}/{epochs} | Loss: {_loss:.5f} | Accuracy: {_acc:.4f}% | F1 Score: {_f1:.4f} | AUROC: {_roc:.4f} | Time: {time.time() - t}')
            
            
        avg_loss = torch.mean(torch.tensor(self.hist['train_loss']))
        avg_acc = torch.mean(torch.tensor(self.hist['train_acc']))
        avg_f1 = torch.mean(torch.tensor(self.hist['train_f1']))
        avg_auroc = torch.mean(torch.tensor(self.hist['train_auroc']))

        print(f'Epoch Training Time: {(time.time() - train_time)/60} min | Avg Loss: {avg_loss:.5f} | Avg Accuracy: {avg_acc:.4f}% | Avg F1 Score: {avg_f1:.4f} | Avg AUROC:{avg_auroc:.4f}')
        
        
        #testing------------------
    
    def test(self, test_loader):
        test_time = time.time()
        test_loss = self.loss_meter()
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(self.device)
        test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=2, average='macro').to(self.device)       
        test_auroc = torchmetrics.AUROC(task="multiclass", num_classes=2).to(self.device)
        
        for idx, batch in enumerate(test_loader):
            
            self.model.eval()
            with torch.no_grad():
                features = batch['X'].to(self.device)
                #targets = batch['y'].type(torch.cuda.LongTensor).to(self.device)
                targets = batch['y'].type(torch.cuda.LongTensor).to(self.device)
                
                org = batch['org']
                print(org)
                
                logits, probs = self.model(features, org)
                loss = self.criterion(logits, targets)
            
                test_loss.update(loss.detach().item())
                test_acc.update(probs.detach(), targets)
                test_f1.update(probs.detach(), targets)
                test_auroc.update(probs.detach(), targets)
                print('------BATCH ENDING-------')

            _loss = test_loss.avg
            _acc = test_acc.compute()
            _f1 = test_f1.compute()
            _roc = test_auroc.compute()

            wandb.log({'test loss': _loss,
                      'test acc': _acc,
                      'test f1_score': _f1,
                      'test AUROC': _roc
                     })
    
            test_acc.reset()
            test_f1.reset()
            test_auroc.reset()
            
            self.hist['test_loss'].append(_loss)
            self.hist['test_acc'].append(_acc)
            self.hist['test_f1'].append(_f1)
            self.hist['test_auroc'].append(_roc)
            

        avg_loss = torch.mean(torch.tensor(self.hist['test_loss']))
        avg_acc = torch.mean(torch.tensor(self.hist['test_acc']))
        avg_f1 = torch.mean(torch.tensor(self.hist['test_f1']))
        avg_auroc = torch.mean(torch.tensor(self.hist['test_auroc']))

        print(f"Testing Time: {(time.time() - test_time)/60:.2f} min | Avg Loss: {avg_loss:.5f} | Avg Accuracy: {avg_acc:.2f}% | Avg F1 Score: {avg_f1:.4f} | Avg AUROC: {avg_auroc:.4f}")
        
        return avg_loss, avg_acc, avg_f1, avg_auroc
                        
                
    def save_model(self, n_epoch, save_path):
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best__score": self.best_valid_score,
                    "best_f1_score": self.best_f_score,
                    "n_epoch": n_epoch,
                },
                save_path,
            )