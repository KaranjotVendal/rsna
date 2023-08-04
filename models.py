class RecNet(nn.Module):
    def __init_(self):
        super().__init__()
        
        #self.CNN = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.CNN = timm.create_model('resnet50', pretrained=True, num_classes=0, in_chans=1)
        in_feature = self.CNN.fc.in_feature
        
        self.rnn = nn.GRU(input_size=in_feature, hidden_size=64, batch_first= True, bidirectional=False)
        
        self.fc = self.Linear(hidden_size, 32, bias=True)
        self.classifier = self.Linear(32, num_calsses=2, bias=True)

    def forward(self, x, mask_in, mask_dup):
        mask = mask_layer(mask_in, mask_dup)
        
        out = self.CNN(x)
        out = self.RNN(out)
        out = out * mask
        out = self.fc(out)

        logits = self.classifier(out)
        output = F.softmax(logits, dim=1)
        #output = F.softmax(logits) #[prob 0, prob 1]

    def mask_layer(self, mask_in, mask_dup):
        mask_1 = torch.ones(mask_in, 64)
        mask_0 = torch.zeros(mask_dup, 64)
        
        return torch.cat((mask_1, mask_0), 0)