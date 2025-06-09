import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim=128, label_dim=16, output_len=100, output_dim=3):
        super().__init__()
        self.label_emb = nn.Embedding(16, label_dim)  # 16 classes
        self.fc = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_len * output_dim)
        )
        self.output_len = output_len
        self.output_dim = output_dim

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)
        x = torch.cat([z, label_vec], dim=1)
        out = self.fc(x)
        return out.view(-1, self.output_len, self.output_dim)


class Discriminator(nn.Module):
    def __init__(self, label_dim=16, input_len=100, input_dim=3):
        super().__init__()
        self.label_emb = nn.Embedding(16, label_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_len * input_dim + label_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, labels):
        label_vec = self.label_emb(labels)
        x_flat = x.view(x.size(0), -1)
        x = torch.cat([x_flat, label_vec], dim=1)
        return self.fc(x)