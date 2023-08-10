import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import timm


class RACNet(nn.Module):
    def __init__(self, MODEL, num_classes):
        super().__init__()
        
        self.cnn = timm.create_model(MODEL, pretrained=True, num_classes=0, in_chans=1)
        for param in self.cnn.parameters():
            param.requires_grad = False
        in_features = self.cnn(torch.randn(2, 1, 112, 112)).shape[1]
        
        self.rnn = nn.GRU(input_size=in_features, hidden_size=64, batch_first= True, bidirectional=False)
        
        self.fc = nn.Linear(16256, 32, bias=True)
        self.classifier = nn.Linear(32, num_classes, bias=True)

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
        out = out.reshape(batch_size, slices * rnn_features)
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
            dup = 254 - i
            mask_1 = torch.ones(i, 64) # .to(device='cuda')
            mask_0 = torch.zeros(dup, 64) #.to(device='cuda')
            mask = torch.cat((mask_1, mask_0), 0)
            masks.append(mask)

        masks = torch.stack(masks).to(device='cuda')
        return masks