import os
import torch
import shutil
import kagglehub
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

def download_data(data_dir, kaggle_data):
    path = kagglehub.dataset_download(kaggle_data)
    if path != data_dir:
        for item in os.listdir(path):
            shutil.move(os.path.join(path, item), os.path.join(data_dir, item))

class NIHChestDataset(Dataset):
    """Lazy-loading NIH Chest X-ray dataset (avoids loading all images into RAM)."""

    def __init__(self, data_path='data', split='train'):
        Path(data_path).mkdir(exist_ok=True)
        if not os.listdir(data_path):
            download_data(data_path, "nih-chest-xrays/data")

        self.data_path = data_path
        self.split = split

        self.image_dirs = [
            os.path.join(data_path, d, 'images') for d in os.listdir(data_path)
            if d.startswith('images_') and os.path.isdir(os.path.join(data_path, d))
        ]
        self.image_dirs.sort()

        self.my_classes = [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
        ]
        self.label2idx = {c: i for i, c in enumerate(self.my_classes)}

        csv_path = os.path.join(data_path, 'Data_Entry_2017.csv')
        df = pd.read_csv(csv_path)

        with open(os.path.join(data_path, 'test_list.txt'), 'r') as f:
            test_list = {line.strip() for line in f}
        with open(os.path.join(data_path, 'train_val_list.txt'), 'r') as f:
            train_val_list = {line.strip() for line in f}

        if split == 'test':
            df = df[df['Image Index'].isin(test_list)]
        elif split == 'train':
            df = df[df['Image Index'].isin(train_val_list)]
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.samples = []
        print(f"Indexing {split} set ({len(df)} rows)...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Indexing {split}"):
            img_path = None
            for image_dir in self.image_dirs:
                potential_path = os.path.join(image_dir, row['Image Index'])
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            if img_path is None:
                continue

            y = np.zeros(len(self.my_classes), dtype=np.float32)
            for l in row['Finding Labels'].split('|'):
                if l in self.label2idx:
                    y[self.label2idx[l]] = 1.0

            self.samples.append((img_path, y))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

def nih_chest_dataset(data_path='data', split='train', return_labels=False):
    ds = NIHChestDataset(data_path=data_path, split=split)
    if return_labels:
        # Load all data into tensors for evaluation
        images = []
        labels = []
        for img, label in tqdm(ds, desc=f"Loading {split} data"):
            images.append(img)
            labels.append(label)
        X = torch.stack(images)
        y = torch.stack(labels)
        return X, y, ds.my_classes
    return ds

def get_class_distribution(data_path='data', split='train'):
    """Get the count of samples for each class in the dataset."""
    ds = NIHChestDataset(data_path=data_path, split=split)
    
    class_counts = np.zeros(len(ds.my_classes), dtype=np.int32)
    
    for _, label in tqdm(ds, desc=f"Counting {split} class distribution"):
        class_counts += label.numpy().astype(np.int32)
    
    print(f"\n{split.upper()} set class distribution:")
    print("-" * 40)
    for class_name, count in zip(ds.my_classes, class_counts):
        print(f"{class_name:20s}: {count:6d}")
    print("-" * 40)
    print(f"Total samples: {len(ds)}")
    
    return dict(zip(ds.my_classes, class_counts))