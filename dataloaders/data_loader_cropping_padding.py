import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import pad
import matplotlib.pyplot as plt

class CroppedPaddedDataset(Dataset):
    def __init__(self, file_path, img_dir, transform=None):
        df = pd.read_csv(file_path)

        self.img_dir = Path(img_dir)
        self.transform = transforms.ToTensor() if transform is None else transform

        self.samples = []
        for _, row in df.iterrows():
            name = str(row.iloc[0]).strip()
            label = row.iloc[1]
            path = self.img_dir / name

            if path.exists():
                self.samples.append((path, label))

        self.target_w = 400
        self.target_h = 200
        print(f"Image width {self.target_w} and height {self.target_h}")
        print(f"Found {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        w, h = img.size
        new_h = self.target_h
        new_w = int(w * (new_h / h))
        img = img.resize((new_w, new_h))

        img = self.transform(img)
        _, h, w = img.shape

        #If greater than target
        if w > self.target_w:
            start = (w - self.target_w) // 2
            img = img[:, :, start:start + self.target_w]

        #If smaller than target
        elif w < self.target_w:
            pad_w = self.target_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img = pad(img, (pad_left, pad_right, 0, 0), value=0)

        return img, label, path.name