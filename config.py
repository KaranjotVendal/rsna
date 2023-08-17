import torch

class key():
    
    wandb_key = 'a2a7828ed68b3cba08f2703971162138c680b664'

class config():
    DATA_PATH = 'data/reduced_dataset/'
    MODEL = 'Res18GRU'
    BATCH_SIZE = 3
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.0001
    
    NUM_WORKERS = 0
    NUM_CLASSES = 2
    KFOLD= 10

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