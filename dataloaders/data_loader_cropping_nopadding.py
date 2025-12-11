import pandas as pd
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class CroppedDataset(Dataset):
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

        #Target size
        self.target_w = 200      
        self.target_h = 200    
        print(f"Image width {self.target_w} and height {self.target_h}")
        print(f"Found {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")
        w, h = img.size

        #Width 
        if w > self.target_w:
            start_x = (w - self.target_w) // 2
            img = img.crop((start_x, 0, start_x + self.target_w, h))
        elif w < self.target_w:
            img = img.resize((self.target_w, h), Image.Resampling.LANCZOS)

        #Update size
        w, h = img.size

        #Height
        if h > self.target_h:
            start_y = (h - self.target_h) // 2
            img = img.crop((0, start_y, w, start_y + self.target_h))
        elif h < self.target_h:
            img = img.resize((w, self.target_h), Image.Resampling.LANCZOS)

        #Enhancement 
        img = img.filter(ImageFilter.MedianFilter(size=1))
        img = ImageEnhance.Contrast(img).enhance(2)
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2.0,
            percent=180,
            threshold=5
        ))

        #To tensor
        img = self.transform(img)

        return img, label, path.name
