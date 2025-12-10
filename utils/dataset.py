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

def download_data(data_dir, kaggle_data):
    path = kagglehub.dataset_download(kaggle_data)
    if path != data_dir:
        for item in os.listdir(path):
            shutil.move(os.path.join(path, item), os.path.join(data_dir, item))

def read_nih_chest_dataset(data_path):
    Path(data_path).mkdir(exist_ok=True)
    if not os.listdir(data_path):
        download_data(data_path, "nih-chest-xrays/data")

    image_dirs = [
        os.path.join(data_path, d, 'images') for d in os.listdir(data_path)
        if d.startswith('images_') and os.path.isdir(os.path.join(data_path, d))
    ]
    image_dirs.sort()

    csv_path = os.path.join(data_path, 'Data_Entry_2017.csv')
    my_classes = [
        'Atelectasis',
        'Consolidation',
        'Infiltration',
        'Pneumothorax',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Effusion',
        'Pneumonia',
        'Pleural_thickening',
        'Cardiomegaly',
        'Nodule',
        'Mass',
        'Hernia'
    ]
    label2idx = {c: i for i, c in enumerate(my_classes)}
    num_classes = len(my_classes)

    df = pd.read_csv(csv_path)

    def encode_labels(label_str):
        y = np.zeros(num_classes, dtype=np.float32)
        for l in label_str.split('|'):
            if l in label2idx:
                y[label2idx[l]] = 1.0
        return y
    df['encoded_labels'] = df['Finding Labels'].apply(encode_labels)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    X = []
    Y = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = None
        for image_dir in image_dirs:
            potential_path = os.path.join(image_dir, row['Image Index'])
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        if img_path is None:
            continue
        img = Image.open(img_path).convert('L')
        img = transform(img)
        img = img.squeeze(0)

        X.append(img)
        Y.append(torch.tensor(row['encoded_labels'], dtype=torch.float32))

    X = torch.stack(X)
    Y = torch.stack(Y)

    test_list = set()
    with open(os.path.join(data_path, 'test_list.txt'), 'r') as f:
        test_list = {line.strip() for line in f}
    train_val_list = set()
    with open(os.path.join(data_path, 'train_val_list.txt'), 'r') as f:
        train_val_list = {line.strip() for line in f}

    image_names = df['Image Index'].values
    test_indices = [i for i, fname in enumerate(image_names) if fname in test_list]
    train_indices = [i for i, fname in enumerate(image_names) if fname in train_val_list]

    X_test = X[test_indices]
    Y_test = Y[test_indices]
    X_train = X[train_indices]
    Y_train = Y[train_indices]

    torch.save({'X': X_test, 'Y': Y_test, 'labels': my_classes}, os.path.join(data_path, 'nih_chest_xray_test.pt'))
    torch.save({'X': X_train, 'Y': Y_train, 'labels': my_classes}, os.path.join(data_path, 'nih_chest_xray_train.pt'))

def nih_chest_dataset(data_path='data', split='train', return_labels=False):
    Path(data_path).mkdir(exist_ok=True)
    if not os.listdir(data_path):
        download_data(data_path, "nih-chest-xrays/data")

    test_pt = os.path.join(data_path, 'nih_chest_xray_test.pt')
    train_pt = os.path.join(data_path, 'nih_chest_xray_train.pt')

    if not os.path.exists(test_pt) or not os.path.exists(train_pt):
        read_nih_chest_dataset(data_path)

    if split == 'test':
        data = torch.load(test_pt)
    elif split == 'train':
        data = torch.load(train_pt)
    else:
        raise ValueError("split must be 'train' or 'test'")

    labels = data.get('labels')
    if return_labels:
        return data['X'], data['Y'], labels
    return data['X'], data['Y']
