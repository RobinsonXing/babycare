# model.py
import torch
import torch.nn as nn

class ResidualFC(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out + residual

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, num_classes=8, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)

        self.residual_fc = ResidualFC(dim=hidden_dim * 2, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        final_hidden = torch.cat([hn[-2], hn[-1]], dim=1)

        out = self.residual_fc(final_hidden)
        out = self.classifier(out)
        return out