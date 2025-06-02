import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import random

class BabyMotionDataset(Dataset):
    def __init__(self, 
                 origin_sequence_dir: str,
                 origin_label_dir: str,
                 aug_dirs: list = None,
                 max_len: int = 100,
                 min_len: int = 10,
                 is_train: bool = True,
                 split_ratio: float = 0.8,
                 random_seed: int = 42,
                 transform=None):
        
        self.max_len = max_len
        self.min_len = min_len
        self.is_train = is_train
        self.split_ratio = split_ratio
        self.random_seed = random_seed
        self.transform = transform

        self.sequence_label_pairs = []
        self._load_pairs_from_dir(origin_sequence_dir, origin_label_dir, True)
        if self.is_train and aug_dirs:
            for seq_dir, label_dir in aug_dirs:
                self._load_pairs_from_dir(seq_dir, label_dir, False)

    def _load_pairs_from_dir(self, seq_dir, label_dir, is_origin):
        if not os.path.exists(seq_dir) or not os.path.exists(label_dir):
            print(f"[Warning] Directory not found: {seq_dir} or {label_dir}")
            return
        indices = sorted([
            f[:-4] for f in os.listdir(seq_dir)
            if f.endswith(".csv")
        ])
        if is_origin:
            random.seed(self.random_seed)
            random.shuffle(indices)
            split_idx = int(len(indices) * self.split_ratio)
            if self.is_train:
                indices = indices[:split_idx]
            else:
                indices = indices[split_idx:]
        for idx in indices:
            seq_path = os.path.join(seq_dir, f"{idx}.csv")
            label_path = os.path.join(label_dir, f"{idx}_label.csv")
            self.sequence_label_pairs.append((seq_path, label_path))

    def __len__(self):
        return len(self.sequence_label_pairs)

    def __getitem__(self, idx):
        seq_path, label_path = self.sequence_label_pairs[idx]

        df = pd.read_csv(seq_path)
        sequence = df[["accel_x", "accel_y", "accel_z"]].values
        T = sequence.shape[0]

        if T < self.min_len:
            return self.__getitem__((idx + 1) % len(self))

        if T > self.max_len:
            start = random.randint(0, T - self.max_len)
            sequence = sequence[start:start + self.max_len]

        if self.transform:
            sequence = self.transform(sequence)

        sequence = torch.tensor(sequence, dtype=torch.float32)

        label_df = pd.read_csv(label_path)
        label = label_df.iloc[0]
        action = label["action"]
        metadata = {
            "gender": label["gender"],
            "age": label["age"],
            "dur": float(label["dur"]) if "dur" in label else T
        }

        return sequence, action, metadata