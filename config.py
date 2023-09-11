import torch
import os

class key():
    
    wandb_key = 'a2a7828ed68b3cba08f2703971162138c680b664'

class config():
    DATA_PATH = 'data/reduced_dataset/'
    MODEL = 'ConvxLSTM'
    RNN = 64 #no of hidden units
    FC = 32   #no of units
    NUM_CLASSES = 2
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.0001
    
    NUM_WORKERS = 0
    KFOLD= 5

    WANDB = False
    CHKPT = False
    TRANSFORM = False
    USE_ft_convnext = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
#mod = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    MOD = 'FLAIR'
    if MOD == 'FLAIR':
        N_SLICES = 250 #254
    elif MOD == 'T1wCE':
        N_SLICES = 250 #203
    elif MOD == 'T1w':
        N_SLICES = 250 #203
    elif MOD == 'T2w':
        N_SLICES = 300 #250


    IMG_SIZE = 112

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")