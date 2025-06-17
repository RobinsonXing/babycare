import torch
import torch.nn as nn
import torch.nn.functional as F

# class Generator(nn.Module):
#     def __init__(self, latent_dim=100, embedding_dim=32, num_classes=16, hidden_dim=128, output_len=100, output_dim=3):
#         super().__init__()
#         self.label_emb = nn.Embedding(num_classes, embedding_dim)
#         self.fc = nn.Linear(latent_dim + embedding_dim, output_len * hidden_dim)
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
#         self.to_output = nn.Linear(hidden_dim, output_dim)
#         self.output_len = output_len

#     def forward(self, z, labels):
#         label_vec = self.label_emb(labels)
#         x = torch.cat([z, label_vec], dim=1)
#         x = self.fc(x)
#         x = x.view(-1, self.output_len, x.size(-1) // self.output_len)
#         out, _ = self.lstm(x)
#         out = self.to_output(out)
#         return out

class Generator(nn.Module):
    def __init__(self, latent_dim=100, embedding_dim=32, num_classes=16,
                 seq_len=100, output_dim=3, hidden_dim=256, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.input_fc = nn.Linear(latent_dim + embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, labels):
        label_vec = self.label_emb(labels)  # (B, embedding_dim)
        gen_input = torch.cat([z, label_vec], dim=1)  # (B, latent+embed)
        h0 = torch.tanh(self.input_fc(gen_input))  # (B, hidden_dim)

        h0 = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)  # (num_layers, B, hidden_dim)
        c0 = torch.zeros_like(h0)

        lstm_input = h0[-1].unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))  # (B, seq_len, hidden_dim)
        out = self.output_fc(lstm_out)  # (B, seq_len, output_dim)

        return out


class Discriminator(nn.Module):
    def __init__(self, embedding_dim=32, num_classes=16, input_dim=3, hidden_dim=128):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.lstm = nn.LSTM(input_dim + embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # No sigmoid in WGAN

    def forward(self, x, labels):
        label_vec = self.label_emb(labels).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, label_vec], dim=2)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden).squeeze(1)
