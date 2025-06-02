from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch, label2idx):
    sequences, actions, metadata = zip(*batch)

    # 每条序列长度（未填充前）
    lengths = [len(seq) for seq in sequences]

    # 对序列进行 padding，shape → (B, T_max, 3)
    padded_sequences = pad_sequence(sequences, batch_first=True)  # shape: (B, T, 3)

    # 标签转 index
    labels = torch.tensor([label2idx[action] for action in actions], dtype=torch.long)

    return padded_sequences, lengths, labels