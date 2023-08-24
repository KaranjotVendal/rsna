import torch

class key():
    
    wandb_key = 'abc'

class config():
    DATA_PATH = 'data/reduced_dataset/'
    MODEL = 'Res50GRU'
    RNN = 64
    FC = 32   
    NUM_CLASSES = 2
    
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.00005
    
    NUM_WORKERS = 0
    KFOLD= 5

    WANDB = False
    CHKPT = False
    
#mod = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
    MOD = 'FLAIR'
    if MOD == 'FLAIR':
        N_SLICES = 254
    elif MOD == 'T1wCE':
        N_SLICES = 203
    elif MOD == 'T1w':
        N_SLICES = 203
    elif MOD == 'T2w':
        N_SLICES = 250


    IMG_SIZE = 112

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")