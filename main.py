import torch
import numpy as np
import lsnet.model.lsnet as lsnet
# mac無法使用SKA, 改用普通的 Conv
# import lsnet.model.lsnet_mac as lsnet
from huggingface_hub import hf_hub_download
from keras.datasets import mnist
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import trange, tqdm


NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = 'cuda' # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", DEVICE)


# -----------------------------
# 1. 載入 MNIST
# -----------------------------
print('下載dataset')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('train data =', len(x_train))
print('test data =', len(x_test))

# 轉為 float32 並正規化
print('正規化')
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# MNIST 只有 1 channel → LSNet 預設 3 channels，需要複製
x_train = np.stack([x_train]*3, axis=1)  # shape: (N,3,28,28)
x_test = np.stack([x_test]*3, axis=1)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_ds = TensorDataset(torch.tensor(x_train), y_train)
test_ds = TensorDataset(torch.tensor(x_test), y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print('資料處理 FIN.')



# -----------------------------
# 2. 建立模型
# -----------------------------
print('載入預訓練權重')
ckpt_path = hf_hub_download("jameslahm/lsnet_t", "pytorch_model.bin")
state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
# NUM_CLASSES 預設 1000, 若 NUM_CLASSES != 1000 則要移除最後一層權重
state.pop("head.l.weight", None)
state.pop("head.l.bias", None)
print('載入完成')

print('建立模型: lsnet_t')
model = lsnet.lsnet_t(
    num_classes=NUM_CLASSES,
    distillation=False,
    pretrained=False,
    frozen_stages=0
)
model.load_state_dict(state, strict=False)
print("LSNet loaded OK!")



# -----------------------------
# 3. Loss & Optimizer
# -----------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




# -----------------------------
# 4. 訓練
# -----------------------------

# train_length = 5
train_length = len(train_loader)
print(f'train data length = {train_length} / {len(train_loader)}')

# EPOCHS = 1
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")
    model.train()
    total_loss = 0
    
    pbar = tqdm(total=train_length)
    for i, (imgs, labels) in enumerate(train_loader):
        if i >= train_length:
            break
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        # imgs = imgs.repeat(1, 3, 1, 1)  # (B,1,H,W) → (B,3,H,W)
        # MNIST 28x28 → LSNet 預期 224x224, 放大
        imgs = torch.nn.functional.interpolate(imgs, size=(224,224), mode='bilinear')
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.update(1)
    pbar.close()
        
    print(f"Epoch {epoch+1}: Loss = {total_loss/train_length:.4f}")



# -----------------------------
# 5. 測試
# -----------------------------
model.eval()
correct = 0
total = 0

# test_length = 10
test_length = len(test_loader)
print(f'test data length = {test_length} / {len(test_loader)}')

pbar = tqdm(total=test_length)
with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        if i >= test_length:
            break
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs = torch.nn.functional.interpolate(imgs, size=(224,224), mode='bilinear')
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.update(1)
        
pbar.close()

print(f"Test Accuracy: {correct/total:.4f}")










