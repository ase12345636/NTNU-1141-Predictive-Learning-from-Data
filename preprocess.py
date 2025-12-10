#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# In[ ]:


# X : (N, 224, 224) tensor, N 張灰階影像
# Y : (N, num_classes) tensor, multi-hot 編碼，NO FINDING 為全 0 向量
# RESIZE FROM 1024x1024 to 224x224
# NORMALIZE: 將影像像素值從 [0, 255] 縮放到 [0.0, 1.0] (除以 255)


image_dir = 'base_img'
csv_path = 'Data_Entry_2017.csv'
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
    transforms.ToTensor(),              # 自動除以 255
])

#create x,y
X = []
Y = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(image_dir, row['Image Index'])
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert('L')   # 灰階
    img = transform(img)                      # (1, 224, 224)
    img = img.squeeze(0)                      # (224, 224)

    X.append(img)
    Y.append(torch.tensor(row['encoded_labels'], dtype=torch.float32))

X = torch.stack(X)   # (N, 224, 224)
Y = torch.stack(Y)   # (N, num_classes)



print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("\n第一張影像 x（2D tensor）:")
print(X[0])
print("x min / max:", X[0].min().item(), X[0].max().item())

print("\n第一筆 label y（multi-hot）:")
print(Y[0])

print("\n該樣本對應的疾病:")
for i, v in enumerate(Y[0]):
    if v == 1:
        print("-", my_classes[i])


# In[ ]:


# 儲存 X, Y 以及類別標籤
torch.save(
    {
        'X': X,
        'Y': Y,
        'labels': my_classes,
    },
    'nih_chest_xray_xy.pt'
)


# In[ ]:




