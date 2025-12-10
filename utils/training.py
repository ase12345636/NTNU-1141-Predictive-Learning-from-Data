import torch
import numpy as np
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    
    for imgs, labels in tqdm(train_loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        with torch.no_grad():
            train_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            train_labels.append(labels.cpu())
    
    train_loss /= len(train_loader)
    train_preds = torch.cat(train_preds).numpy()
    train_labels = torch.cat(train_labels).numpy()
    train_acc = 1 - np.mean(np.abs(train_preds - train_labels))
    
    return train_loss, train_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            val_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            val_labels.append(labels.cpu())
    
    val_loss /= len(val_loader)
    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    val_acc = 1 - np.mean(np.abs(val_preds - val_labels))
    
    return val_loss, val_acc

def test_model(model, test_loader, criterion, device):
    """Test model on test set and measure per-image inference time (ms)."""
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
    test_acc = 1 - np.mean(np.abs(test_preds - test_labels))
    avg_ms = float(np.mean(timings) * 1000.0) if timings else None
    
    return test_loss, test_acc, test_preds, test_labels, avg_ms
