import time
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score

def train_epoch_combined(model, train_loader, criterion, optimizer, device, scaler: GradScaler | None = None):
    """Train for one epoch with both accuracy and F1 score evaluation."""
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    scaler = scaler or GradScaler(enabled=device.startswith('cuda'))
    
    for imgs, labels in tqdm(train_loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.startswith('cuda')):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        
        with torch.no_grad():
            train_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            train_labels.append(labels.cpu())
    
    train_loss /= len(train_loader)
    train_preds = torch.cat(train_preds).numpy()
    train_labels = torch.cat(train_labels).numpy()
    
    # Compute both accuracy and F1 scores
    train_acc = 1 - np.mean(np.abs(train_preds - train_labels))
    train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    train_f1_weighted = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
    
    return train_loss, train_acc, train_f1_macro, train_f1_weighted

def validate_epoch_combined(model, val_loader, criterion, device):
    """Validate for one epoch with both accuracy and F1 score evaluation."""
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=device.startswith('cuda')):
                logits = model(imgs)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            
            val_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            val_labels.append(labels.cpu())
    
    val_loss /= len(val_loader)
    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    
    # Compute both accuracy and F1 scores
    val_acc = 1 - np.mean(np.abs(val_preds - val_labels))
    val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    
    return val_loss, val_acc, val_f1_macro, val_f1_weighted

def test_model_combined(model, test_loader, criterion, device):
    """Test model with both accuracy and F1 score evaluation and per-image inference time."""
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    timings = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Testing'):
            imgs, labels = imgs.to(device), labels.to(device)
            start = time.perf_counter()
            logits = model(imgs)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) / imgs.size(0))

            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            test_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            test_labels.append(labels.cpu())
    
    test_loss /= len(test_loader)
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    
    # Compute both accuracy and F1 scores
    test_acc = 1 - np.mean(np.abs(test_preds - test_labels))
    test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1_per_class = f1_score(test_labels, test_preds, average=None, zero_division=0)
    
    avg_ms = float(np.mean(timings) * 1000.0) if timings else None
    
    return test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_f1_per_class, avg_ms
