import torch
import torch.nn as nn
from torchvision import models


class EventDetector(nn.Module):
    def __init__(self, pretrain, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        #load efficientnet backbone with pretrained weights
        net = models.efficientnet_b0(pretrained=pretrain)
        # extract feature layers only 
        self.cnn = net.features

        # using a dummy forward pass to avoid hardcoding architecture specific values
        with torch.no_grad():
            dummy = torch.randn(1, 3, 160, 160)
            out = self.cnn(dummy)
            self.feature_dim = out.shape[1]
            print(f'EfficientNet-B0 feature dim: {self.feature_dim}')


        # Bidirectional LSTM for temporal modelling
        self.rnn = nn.LSTM(self.feature_dim,
                           lstm_hidden,
                           lstm_layers,
                           batch_first=True,
                           bidirectional=bidirectional)
        
        # 9 output classes, the 8 events and non-event class
        if self.bidirectional:
            self.lin = nn.Linear(2 * lstm_hidden, 9)
        else:
            self.lin = nn.Linear(lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)


    def init_hidden(self, batch_size, device):
        num_directions = 2 if self.bidirectional else 1
        h = torch.zeros(self.lstm_layers * num_directions, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(self.lstm_layers * num_directions, batch_size, self.lstm_hidden, device=device)
        return (h, c)


    def forward(self, x, lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        # reinitialise hidden state for each forward pass
        self.hidden = self.init_hidden(batch_size, x.device)


        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = torch.mean(c_out, dim=[2, 3])
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        out = self.lin(r_out)
        out = out.view(batch_size * timesteps, 9)

        return out