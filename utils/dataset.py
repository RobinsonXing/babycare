import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class BabyMotionDataset(Dataset):
    def __init__(self, 
                 origin_dir: str,
                 aug_dirs: list[str],
                 max_len: int = 100,
                 min_len: int = 10,
                 is_train: bool = True,
                 transform=None):
        
        self.max_len = max_len
        self.min_len = min_len
        self.transform = transform

        origin_seq_dir, origin_label_dir = self._find_pair_dirs(origin_dir)
        id_txt = os.path.join(origin_dir, "train.txt" if is_train else "val.txt")
        if not os.path.exists(id_txt):
            raise FileNotFoundError(f"{id_txt} not found.")
        with open(id_txt, "r") as f:
            origin_indices = [line.strip() for line in f if line.strip()]

        self.sequence_label_pairs = []
        self._load_pairs_from_dir(origin_seq_dir, origin_label_dir, origin_indices)
        if is_train and aug_dirs:
            aug_pair_dirs = [(self._find_pair_dirs(aug_dir)) for aug_dir in aug_dirs]
            for aug_seq_dir, aug_label_dir in aug_pair_dirs:
                aug_indices = sorted([
                    fname[:-4] for fname in os.listdir(aug_seq_dir)
                    if fname.endswith(".csv")
                ])
                self._load_pairs_from_dir(aug_seq_dir, aug_label_dir, aug_indices)

    def _find_pair_dirs(self, base_dir: str):
        sequence_dir = os.path.join(base_dir, "sequence")
        label_dir = os.path.join(base_dir, "label")
        return sequence_dir, label_dir

    def _load_pairs_from_dir(self, seq_dir, label_dir, indices):
        for idx in indices:
            seq_path = os.path.join(seq_dir, f"{idx}.csv")
            label_path = os.path.join(label_dir, f"{idx}_label.csv")
            if os.path.exists(seq_path) and os.path.exists(label_path):
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
            sequence = sequence[:self.max_len]

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