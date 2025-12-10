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

    torch.save(
    {
        'X': X,
        'Y': Y,
        'labels': my_classes,
    },
    os.path.join(data_path, 'nih_chest_xray.pt')
    )

def nih_chest_dataset(data_path = 'data'):
    if not os.path.exists(os.path.join(data_path, 'nih_chest_xray.pt')):
        read_nih_chest_dataset(data_path)

    data = torch.load(os.path.join(data_path, 'nih_chest_xray.pt'))
    X = data['X']
    Y = data['Y']
    my_classes = data['labels']
    return X, Y, my_classes

if __name__ == '__main__':
    X, y = nih_chest_dataset()

    # 檢查資料集的基本資訊
    print("=" * 50)
    print("資料集基本資訊")
    print("=" * 50)
    print(f"總共載入的影像數量: {len(X)}")
    print(f"影像張量形狀: {X.shape}")
    print(f"標籤張量形狀: {y.shape}")
    print(f"影像數據類型: {X.dtype}")
    print(f"標籤數據類型: {y.dtype}")
    print()
    
    # 檢查前5筆資料
    print("=" * 50)
    print("前5筆資料檢查")
    print("=" * 50)
    
    my_classes = [
        'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
        'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
        'Pleural_thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
    ]
    
    for i in range(min(5, len(X))):
        print(f"\n第 {i+1} 筆資料:")
        print(f"  影像形狀: {X[i].shape}")
        print(f"  影像值範圍: [{X[i].min().item():.4f}, {X[i].max().item():.4f}]")
        print(f"  標籤向量: {y[i].numpy()}")
        
        # 顯示有哪些疾病標籤
        positive_labels = [my_classes[j] for j in range(len(my_classes)) if y[i][j] == 1.0]
        if positive_labels:
            print(f"  檢測到的疾病: {', '.join(positive_labels)}")
        else:
            print(f"  檢測到的疾病: 無疾病 (正常)")
    
    print("\n" + "=" * 50)