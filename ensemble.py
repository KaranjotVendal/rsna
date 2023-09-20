import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import timm

from config import config
from datamodule import RSNAdataset
from utils import save_metrics_to_json


class ConvxLSTM(nn.Module):
    '''ConvNext: pretrained IMAGENET, not trainable
        LSTM: 1 layer, 64 units, unidirectional
        FC: 32 units
        classifier: 2 units'''
    def __init__(self, num_classes, N_SLICES):
        super().__init__()
        
        self.N_SLICES = N_SLICES
        self.cnn = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0, in_chans=1)
        if config.USE_ft_convnext:
            checkpoint = torch.load(f'./data/pretrain_convnext/ConvNext_finetuned_model_best_auroc.pth')
            self.cnn.load_state_dict(checkpoint["model_state_dict"], strict=False)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, config.IMG_SIZE, config.IMG_SIZE)).shape[1]
        
        self.rnn = nn.LSTM(input_size=in_features, hidden_size=config.RNN, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(self.N_SLICES * config.RNN, config.FC, bias=True)
        self.classifier = nn.Linear(config.FC, num_classes, bias=True)        
        
    def forward(self, x, org):
        # x shape: BxSxCxHxW
        batch_size, slices, C, H, W = x.size()
        c_in = x.view(batch_size * slices, C, H, W)
        out = self.cnn(c_in)
        rnn_in = out.view(batch_size, slices, -1)
        out, hd = self.rnn(rnn_in)
        mask = self.mask_layer(org)
        out = out * mask
        batch, slices, rnn_features = out.size()
        out = out.reshape(batch, slices * rnn_features)
        out = F.relu(self.fc(out))
        logits = self.classifier(out)
        output = F.softmax(logits, dim=1)

        return logits, output

    def mask_layer(self, org):
        masks = []
        org = org[0].cpu().numpy()
        for i in org:
            dup = self.N_SLICES - i
            mask_1 = torch.ones(i, config.RNN) # .to(device='cuda')
            mask_0 = torch.zeros(dup, config.RNN) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(config.DEVICE)
        return masks




def majority_voting(predictions):
    final_predictions = []

    for idx, value in enumerate(predictions[0]):
        pred=torch.argmax(value, dim=0).item()
        pred2=torch.argmax(predictions[1][idx], dim=0).item()
        pred3=torch.argmax(predictions[2][idx], dim=0).item()
        pred4=torch.argmax(predictions[3][idx], dim=0).item()
        
        sum_votes = pred + pred2 + pred3 +pred4
            
        if sum_votes >= 3:
            final_prediction = 1
        elif sum_votes == 2:
            final_prediction = random.choice([0, 1])
        else:
            final_prediction = 0

        final_predictions.append(final_prediction)
    
    return final_predictions

def sum_of_probabilities(predictions):
    final_predictions = []

    for idx, value in enumerate(predictions[0]):
        l = []
        l.append(value.numpy())
        l.append(predictions[1][idx].numpy())
        l.append(predictions[2][idx].numpy())
        l.append(predictions[3][idx].numpy())

        final_predictions.append(np.argmax(np.sum(np.array(l), axis=0)))

    return final_predictions

def ensemble_prediction(models, test_loader, approach='majority_voting'):
    all_predictions = []
    all_true_labels = []

    predictions = {}
    mri_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']

    for mri_type in mri_types:
        pred = []
        for e, batch in enumerate(test_loader[mri_type]):
            with torch.no_grad():
                features = batch['X'].to(config.DEVICE)
                targets = batch['y']
                org = batch['org']

                _, probs = models[mri_type](features, org)
                pred.append(probs.cpu())
                if mri_type=='T1w':
                    all_true_labels.extend(targets.numpy().tolist())
        
        predictions[mri_type] = pred

    fused_preds = zip(predictions['FLAIR'],predictions['T1w'],predictions['T1wCE'], predictions['T2w'])

    for i in fused_preds:
        if approach == 'majority_voting':
            final_prediction = majority_voting(list(i))
        elif approach == 'sum_of_probabilities':
            final_prediction = sum_of_probabilities(list(i))

        all_predictions.append(final_prediction)
        
    return np.concatenate(all_predictions), all_true_labels

def evaluate_fold(fold, data_loader):
    # Load the best models for each experiment
    models = {}
    for experiment, modality in {3:'FLAIR', 7:'T1w',8:'T1wCE', 9:'T2w'}.items(): #8:'T1wCE'
        if modality == 'T2w':
            n_slices=300
        else:
            n_slices=250
        
        model = ConvxLSTM(config.NUM_CLASSES, N_SLICES=n_slices).to(config.DEVICE)
        checkpoint = torch.load(f'.\experiments2.0\experiment {experiment}\weights\ConvxLSTM_model_{modality}_{fold}.pth')
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.DEVICE)
        model.eval()
        models[modality] = model

    # Approach 1: Majority Voting
    predictions_mv, true_labels_mv = ensemble_prediction(models, data_loader, 'majority_voting')
    f1_mv = f1_score(true_labels_mv, predictions_mv, average='macro')
    auroc_mv = roc_auc_score(true_labels_mv, predictions_mv)
    acc_mv = accuracy_score(true_labels_mv, predictions_mv)

    # Approach 2: Sum of Probabilities
    predictions_sp, true_labels_sp = ensemble_prediction(models, data_loader, 'sum_of_probabilities')
    f1_sp = f1_score(true_labels_sp, predictions_sp)
    auroc_sp = roc_auc_score(true_labels_sp, predictions_sp)
    acc_sp = accuracy_score(true_labels_sp, predictions_sp)

    return (f1_mv, auroc_mv, acc_mv), (f1_sp, auroc_sp, acc_sp)

def main():
    
    f1_mv_scores = []
    auroc_mv_scores = []
    acc_mv_scores = []

    f1_sp_scores = []
    auroc_sp_scores = []
    acc_sp_scores = []

    '''metrics = {
        'mv': {'acc': [], 'f1': [], 'auroc': [], 'f1_spread': [], 'auroc_spread': [], 'acc_spread': [],},
        'sp': {'acc': [], 'f1': [], 'auroc': [], 'f1_spread': [], 'auroc_spread': [], 'acc_spread': [],}
        }'''
    metrics = {
        'mv': {},
        'sp': {}
        }
    

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

    mri_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):  
    
        print(f'--------------FOLD:{fold}-----------------------') 
        
        xtest = X[test_idx]
        ytest = Y[test_idx]

        datasets = {}
        loaders = {}

        for mri_type in mri_types:
            if mri_type == "T2w":
                n_slices = 300
            else:
                n_slices = 250
            
            datasets[mri_type] = RSNAdataset(
                config.DATA_PATH,
                xtest,  
                ytest,
                n_slices=n_slices,
                img_size=config.IMG_SIZE,
                type=mri_type,
                transform=None
            )
            
            loaders[mri_type] = DataLoader(
                datasets[mri_type],    
                batch_size=6,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
            )     

        (f1_mv, auroc_mv, acc_mv), (f1_sp, auroc_sp, acc_sp) = evaluate_fold(fold, loaders)
        
        f1_mv_scores.append(f1_mv)
        auroc_mv_scores.append(auroc_mv)
        acc_mv_scores.append(acc_mv)

        f1_sp_scores.append(f1_sp)
        auroc_sp_scores.append(auroc_sp)
        acc_sp_scores.append(acc_sp)

    metrics['mv']['f1'] = f1_mv_scores
    metrics['mv']['acc'] = acc_mv_scores
    metrics['mv']['auroc'] = auroc_mv_scores

    avg_f1_mv = np.mean(f1_mv_scores)
    f1_max_mv = np.max(f1_mv_scores) - avg_f1_mv
    f1_min_mv = avg_f1_mv - np.min(f1_mv_scores)
    spread_f1_mv = [f1_min_mv,f1_max_mv]

    avg_auroc_mv = np.mean(auroc_mv_scores)
    auroc_max_mv = np.max(auroc_mv_scores) - avg_auroc_mv
    auroc_min_mv = avg_auroc_mv - np.min(auroc_mv_scores)
    spread_auroc_mv = [auroc_min_mv,auroc_max_mv]

    avg_acc_mv = np.mean(acc_mv_scores)
    acc_max_mv = np.max(acc_mv_scores) - avg_acc_mv
    acc_min_mv = avg_acc_mv - np.min(acc_mv_scores)
    spread_acc_mv = [acc_min_mv, acc_max_mv]

    metrics['mv']['f1_spread'] = spread_f1_mv
    metrics['mv']['auroc_spread'] = spread_auroc_mv
    metrics['mv']['acc_spread'] = spread_acc_mv


    metrics['sp']['f1'] = f1_sp_scores
    metrics['sp']['acc'] = acc_sp_scores
    metrics['sp']['auroc'] = auroc_sp_scores

    avg_f1_sp = np.mean(f1_sp_scores)
    f1_max_sp = np.max(f1_sp_scores) - avg_f1_sp
    f1_min_sp = avg_f1_sp - np.min(f1_sp_scores)
    spread_f1_sp = [f1_min_sp,f1_max_sp]

    avg_auroc_sp = np.mean(auroc_sp_scores) 
    auroc_max_sp = np.max(auroc_sp_scores) - avg_auroc_sp
    auroc_min_sp = avg_auroc_sp - np.min(auroc_sp_scores)
    spread_auroc_sp = [auroc_min_sp,auroc_max_sp]

    avg_acc_sp = np.mean(acc_sp_scores)
    acc_max_sp = np.max(acc_sp_scores) - avg_acc_sp
    acc_min_sp = avg_acc_sp - np.min(acc_sp_scores)
    spread_acc_sp = [acc_min_sp,acc_max_sp]

    metrics['sp']['f1_spread'] = spread_f1_sp
    metrics['sp']['auroc_spread'] = spread_auroc_sp
    metrics['sp']['acc_spread'] = spread_acc_sp

    json_path = save_metrics_to_json(metrics, 'ensemble method')
    
    print("Average F1 (Majority Voting):", avg_f1_mv, "+-", spread_f1_mv)
    print("Average AUROC (Majority Voting):", avg_auroc_mv, "+-", spread_auroc_mv)
    print("Average Acc (Majority Voting):", avg_acc_mv, "+-", spread_acc_mv)
        
    print("Average F1 (Sum of Probabilities):", avg_f1_sp, "+-", spread_f1_sp)
    print("Average AUROC (Sum of Probabilities):", avg_auroc_sp, "+-", spread_auroc_sp)
    print("Average Acc (sum of Probabilities):", avg_acc_sp, "+-", spread_acc_sp)
    

if __name__ == '__main__':
    main()