from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch, label2idx):
    sequences, actions, metadata = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)  # shape: (B, T, 3)
    labels = torch.tensor([label2idx[action] for action in actions], dtype=torch.long)
    return padded_sequences, lengths, labels