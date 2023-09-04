import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import config


class Res18GRU(nn.Module):
    '''RESNET18: pretrained IMAGENET, not trainable
        GRU: 1 layer, 64 units, unidirectional
        FC: 32 units
        classifier: 2 units'''

    def __init__(self, num_classes):
        super().__init__()
        
        #self.cnn = timm.create_model('resnet18', pretrained=True, num_classes=0, in_chans=1)
        self.cnn = timm.create_model('resnet18.fb_swsl_ig1b_ft_in1k', pretrained=True, num_classes=0, in_chans=1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, 112, 112)).shape[1]
        
        self.rnn = nn.GRU(input_size=in_features, hidden_size=config.RNN, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(config.N_SLICES * config.RNN, config.FC, bias=True)
        self.classifier = nn.Linear(config.FC, num_classes, bias=True)        
        
    def forward(self, x, org):
        # x shape: BxSxCxHxW
        batch_size, slices, C, H, W = x.size()
        c_in = x.view(batch_size * slices, C, H, W)
        #print('reshape input', c_in.shape)
        
        out = self.cnn(c_in)
        #print('CNN ouput', out.shape)
        
        rnn_in = out.view(batch_size, slices, -1)
        #print('reshaped rnn_in', rnn_in.shape)
        out, hd = self.rnn(rnn_in)
        #out =F.relu(self.RNN(out))

        #print('RNN ouput', out.shape)
        mask = self.mask_layer(org)
        out = out * mask
        #print('mask ouput', out.shape)
        
        batch, slices, rnn_features = out.size()
        out = out.reshape(batch, slices * rnn_features)
        #print('reshaped masked output', out.shape)
        
        out = F.relu(self.fc(out))
        #print('fc ouput', out.shape)
        
        logits = self.classifier(out)
        #print('logits', logits.shape)
        
        output = F.softmax(logits, dim=1)
        
        #print('classifier ouput', logits.shape)
        #[prob 0, prob 1]

        return logits, output

    def mask_layer(self, org):
        masks = []
        org = org[0].cpu().numpy()
        for i in org:
            #print(i)
            dup = config.N_SLICES - i
            mask_1 = torch.ones(i, config.RNN) # .to(device='cuda')
            mask_0 = torch.zeros(dup, config.RNN) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(config.DEVICE)
        return masks
    





class Res50GRU(nn.Module):
    '''RESNET50: pretrained IMAGENET, not trainable
        GRU: 1 layer, 64 units, unidirectional
        FC: 32 units
        classifier: 2 units'''

    def __init__(self, num_classes):
        super().__init__()
        
        self.cnn = timm.create_model('resnet50d.ra2_in1k', pretrained=True, num_classes=0, in_chans=1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, 112, 112)).shape[1]
        
        self.rnn = nn.GRU(input_size=in_features, hidden_size=config.RNN, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(config.N_SLICES * config.RNN, config.FC, bias=True)
        self.classifier = nn.Linear(config.FC, num_classes, bias=True)        
        
    def forward(self, x, org):
        # x shape: BxSxCxHxW
        batch_size, slices, C, H, W = x.size()
        c_in = x.view(batch_size * slices, C, H, W)
        #print('reshape input', c_in.shape)
        
        out = self.cnn(c_in)
        #print('CNN ouput', out.shape)
        
        rnn_in = out.view(batch_size, slices, -1)
        #print('reshaped rnn_in', rnn_in.shape)
        out, hd = self.rnn(rnn_in)
        #out =F.relu(self.RNN(out))

        #print('RNN ouput', out.shape)
        mask = self.mask_layer(org)
        out = out * mask
        #print('mask ouput', out.shape)
        
        batch, slices, rnn_features = out.size()
        out = out.reshape(batch, slices * rnn_features)
        #print('reshaped masked output', out.shape)
        
        out = F.relu(self.fc(out))
        #print('fc ouput', out.shape)
        
        logits = self.classifier(out)
        #print('logits', logits.shape)
        
        output = F.softmax(logits, dim=1)
        
        #print('classifier ouput', logits.shape)
        #[prob 0, prob 1]

        return logits, output

    def mask_layer(self, org):
        masks = []
        org = org[0].cpu().numpy()
        for i in org:
            #print(i)
            dup = config.RNN - i
            mask_1 = torch.ones(i, config.RNN) # .to(device='cuda')
            mask_0 = torch.zeros(dup, config.RNN) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(config.DEVICE)
        return masks
    


class Res18LSTM(nn.Module):
    '''RESNET18: pretrained IMAGENET, not trainable
        LSTM: 1 layer, 64 units, unidirectional
        FC: 32 units
        classifier: 2 units'''
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.cnn = timm.create_model('resnet18', pretrained=True, num_classes=0, in_chans=1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, 112, 112)).shape[1]
        
        self.rnn = nn.GRU(input_size=in_features, hidden_size=config.RNN, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(config.N_SLICES * config.RNN, config.FC, bias=True)
        self.classifier = nn.Linear(config.FC, num_classes, bias=True)        
        
    def forward(self, x, org):
        # x shape: BxSxCxHxW
        batch_size, slices, C, H, W = x.size()
        c_in = x.view(batch_size * slices, C, H, W)
        #print('reshape input', c_in.shape)
        
        out = self.cnn(c_in)
        #print('CNN ouput', out.shape)
        
        rnn_in = out.view(batch_size, slices, -1)
        #print('reshaped rnn_in', rnn_in.shape)
        out, hd = self.rnn(rnn_in)
        #out =F.relu(self.RNN(out))

        #print('RNN ouput', out.shape)
        mask = self.mask_layer(org)
        out = out * mask
        #print('mask ouput', out.shape)
        
        batch, slices, rnn_features = out.size()
        out = out.reshape(batch, slices * rnn_features)
        #print('reshaped masked output', out.shape)
        
        out = F.relu(self.fc(out))
        #print('fc ouput', out.shape)
        
        logits = self.classifier(out)
        #print('logits', logits.shape)
        
        output = F.softmax(logits, dim=1)
        
        #print('classifier ouput', logits.shape)
        #[prob 0, prob 1]

        return logits, output

    def mask_layer(self, org):
        masks = []
        org = org[0].cpu().numpy()
        for i in org:
            #print(i)
            dup = config.N_SLICES - i
            mask_1 = torch.ones(i, config.RNN) # .to(device='cuda')
            mask_0 = torch.zeros(dup, config.RNN) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(config.DEVICE)
        return masks
    




class Res50LSTM(nn.Module):
    '''RESNET50: pretrained IMAGENET, not trainable
        LSTM: 1 layer, 64 units, unidirectional
        FC: 32 units
        classifier: 2 units'''

    def __init__(self, num_classes):
        super().__init__()
        
        self.cnn = timm.create_model('resnet50d.ra2_in1k', pretrained=True, num_classes=0, in_chans=1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, 112, 112)).shape[1]
        
        self.rnn = nn.GRU(input_size=in_features, hidden_size=config.RNN, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(config.N_SLICES * config.RNN, config.FC, bias=True)
        self.classifier = nn.Linear(config.FC, num_classes, bias=True)        
        
    def forward(self, x, org):
        # x shape: BxSxCxHxW
        batch_size, slices, C, H, W = x.size()
        c_in = x.view(batch_size * slices, C, H, W)
        #print('reshape input', c_in.shape)
        
        out = self.cnn(c_in)
        #print('CNN ouput', out.shape)
        
        rnn_in = out.view(batch_size, slices, -1)
        #print('reshaped rnn_in', rnn_in.shape)
        out, hd = self.rnn(rnn_in)
        #out =F.relu(self.RNN(out))

        #print('RNN ouput', out.shape)
        mask = self.mask_layer(org)
        out = out * mask
        #print('mask ouput', out.shape)
        
        batch, slices, rnn_features = out.size()
        out = out.reshape(batch, slices * rnn_features)
        #print('reshaped masked output', out.shape)
        
        out = F.relu(self.fc(out))
        #print('fc ouput', out.shape)
        
        logits = self.classifier(out)
        #print('logits', logits.shape)
        
        output = F.softmax(logits, dim=1)
        
        #print('classifier ouput', logits.shape)
        #[prob 0, prob 1]

        return logits, output

    def mask_layer(self, org):
        masks = []
        org = org[0].cpu().numpy()
        for i in org:
            #print(i)
            dup = config.RNN - i
            mask_1 = torch.ones(i, config.RNN) # .to(device='cuda')
            mask_0 = torch.zeros(dup, config.RNN) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(config.DEVICE)
        return masks
    




class ConvxGRU(nn.Module):
    '''ConvNext: pretrained IMAGENET, not trainable
        GRU: 1 layer, 64 units, unidirectional
        FC: 32 units
        classifier: 2 units'''
    
    def __init__(self, num_classes):
        super().__init__()
        
        self.cnn = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0, in_chans=1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, 112, 112)).shape[1]
        
        self.rnn = nn.GRU(input_size=in_features, hidden_size=config.RNN, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(config.N_SLICES * config.BATCH_SIZE, config.FC, bias=True)
        self.classifier = nn.Linear(config.FC, num_classes, bias=True)        
        
    def forward(self, x, org):
        # x shape: BxSxCxHxW
        batch_size, slices, C, H, W = x.size()
        c_in = x.view(batch_size * slices, C, H, W)
        #print('reshape input', c_in.shape)
        
        out = self.cnn(c_in)
        #print('CNN ouput', out.shape)
        
        rnn_in = out.view(batch_size, slices, -1)
        #print('reshaped rnn_in', rnn_in.shape)
        out, hd = self.rnn(rnn_in)
        #out =F.relu(self.RNN(out))

        #print('RNN ouput', out.shape)
        mask = self.mask_layer(org)
        out = out * mask
        #print('mask ouput', out.shape)
        
        batch, slices, rnn_features = out.size()
        out = out.reshape(batch, slices * rnn_features)
        #print('reshaped masked output', out.shape)
        
        out = F.relu(self.fc(out))
        #print('fc ouput', out.shape)
        
        logits = self.classifier(out)
        #print('logits', logits.shape)
        
        output = F.softmax(logits, dim=1)
        
        #print('classifier ouput', logits.shape)
        #[prob 0, prob 1]

        return logits, output

    def mask_layer(self, org):
        masks = []
        org = org[0].cpu().numpy()
        for i in org:
            #print(i)
            dup = config.N_SLICES - i
            mask_1 = torch.ones(i, config.RNN) # .to(device='cuda')
            mask_0 = torch.zeros(dup, config.RNN) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(config.DEVICE)
        return masks
    





class ConvxLSTM(nn.Module):
    '''ConvNext: pretrained IMAGENET, not trainable
        LSTM: 1 layer, 64 units, unidirectional
        FC: 32 units
        classifier: 2 units'''
    def __init__(self, num_classes):
        super().__init__()
        
        self.cnn = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0, in_chans=1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, 112, 112)).shape[1]
        
        self.rnn = nn.GRU(input_size=in_features, hidden_size=config.RNN, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(config.N_SLICES * config.RNN, config.FC, bias=True)
        self.classifier = nn.Linear(config.FC, num_classes, bias=True)        
        
    def forward(self, x, org):
        # x shape: BxSxCxHxW
        batch_size, slices, C, H, W = x.size()
        c_in = x.view(batch_size * slices, C, H, W)
        #print('reshape input', c_in.shape)
        
        out = self.cnn(c_in)
        #print('CNN ouput', out.shape)
        
        rnn_in = out.view(batch_size, slices, -1)
        #print('reshaped rnn_in', rnn_in.shape)
        out, hd = self.rnn(rnn_in)
        #out =F.relu(self.RNN(out))

        #print('RNN ouput', out.shape)
        mask = self.mask_layer(org)
        out = out * mask
        #print('mask ouput', out.shape)
        
        batch, slices, rnn_features = out.size()
        out = out.reshape(batch, slices * rnn_features)
        #print('reshaped masked output', out.shape)
        
        out = F.relu(self.fc(out))
        #print('fc ouput', out.shape)
        
        logits = self.classifier(out)
        #print('logits', logits.shape)
        
        output = F.softmax(logits, dim=1)
        
        #print('classifier ouput', logits.shape)
        #[prob 0, prob 1]

        return logits, output

    def mask_layer(self, org):
        masks = []
        org = org[0].cpu().numpy()
        for i in org:
            dup = config.N_SLICES - i
            mask_1 = torch.ones(i, config.RNN) # .to(device='cuda')
            mask_0 = torch.zeros(dup, config.RNN) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(config.DEVICE)
        return masks