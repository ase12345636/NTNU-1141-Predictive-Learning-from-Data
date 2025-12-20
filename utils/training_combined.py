import time
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

def train_epoch_combined(model, train_loader, criterion, optimizer, device, scaler: GradScaler | None = None, grad_clip_norm: float = None):
    """Train for one epoch with both accuracy and F1 score evaluation.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision training
        grad_clip_norm: Maximum gradient norm for clipping (None = no clipping)
    """
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
        
        # 梯度裁剪
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
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

def train_epoch_combined_auc(model, train_loader, criterion, optimizer, device, scaler: GradScaler | None = None, grad_clip_norm: float = None):
    """Train for one epoch with both accuracy and F1 score evaluation.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision training
        grad_clip_norm: Maximum gradient norm for clipping (None = no clipping)
    """
    model.train()
    train_loss = 0
    train_preds = []
    train_scores = []
    train_labels = []
    scaler = scaler or GradScaler(enabled=device.startswith('cuda'))
    
    for imgs, labels in tqdm(train_loader, desc='Training'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.startswith('cuda')):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        
        with torch.no_grad():
            train_scores.append(logits.cpu())
            train_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            train_labels.append(labels.cpu())
    
    train_loss /= len(train_loader)
    train_preds = torch.cat(train_preds).numpy()
    train_labels = torch.cat(train_labels).numpy()
    train_scores = torch.cat(train_scores).numpy()
    
    # Compute both accuracy and F1 scores
    train_acc = 1 - np.mean(np.abs(train_preds - train_labels))
    train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    train_f1_weighted = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
    train_auc_macro = multilabel_auc_score(train_labels, train_scores, average="macro")
    train_auc_weighted = multilabel_auc_score(train_labels, train_scores, average="weighted")
    train_auprc_macro = multilabel_auprc_score(train_labels, train_scores, average="macro")
    train_auprc_weighted = multilabel_auprc_score(train_labels, train_scores, average="weighted")

    return train_loss, train_acc, train_f1_macro, train_f1_weighted, train_auc_macro, train_auc_weighted, train_auprc_macro, train_auprc_weighted

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

def validate_epoch_combined_auc(model, val_loader, criterion, device):
    """Validate for one epoch with both accuracy and F1 score evaluation."""
    model.eval()
    val_loss = 0
    val_preds = []
    val_scores = []
    val_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(enabled=device.startswith('cuda')):
                logits = model(imgs)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            
            val_scores.append(logits.cpu())
            val_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            val_labels.append(labels.cpu())
    
    val_loss /= len(val_loader)
    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    val_scores = torch.cat(val_scores).numpy()
    
    # Compute both accuracy and F1 scores
    val_acc = 1 - np.mean(np.abs(val_preds - val_labels))
    val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    val_auc_macro = multilabel_auc_score(val_labels, val_scores, average="macro")
    val_auc_weighted = multilabel_auc_score(val_labels, val_scores, average="weighted")
    val_auprc_macro = multilabel_auprc_score(val_labels, val_scores, average="macro")
    val_auprc_weighted = multilabel_auprc_score(val_labels, val_scores, average="weighted")

    return val_loss, val_acc, val_f1_macro, val_f1_weighted, val_auc_macro, val_auc_weighted, val_auprc_macro, val_auprc_weighted

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

def test_model_combined_auc(model, test_loader, criterion, device):
    """Test model with both accuracy and F1 score evaluation and per-image inference time."""
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    test_scores = []
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
            
            test_scores.append(logits.cpu())
            test_preds.append((torch.sigmoid(logits) > 0.5).cpu())
            test_labels.append(labels.cpu())
    
    test_loss /= len(test_loader)
    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_scores = torch.cat(test_scores).numpy()
    
    # Compute both accuracy and F1 scores
    test_acc = 1 - np.mean(np.abs(test_preds - test_labels))
    test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1_per_class = f1_score(test_labels, test_preds, average=None, zero_division=0)
    test_auc_macro = multilabel_auc_score(test_labels, test_scores, average="macro")
    test_auc_weighted = multilabel_auc_score(test_labels, test_scores, average="weighted")
    test_auprc_macro = multilabel_auprc_score(test_labels, test_scores, average="macro")
    test_auprc_weighted = multilabel_auprc_score(test_labels, test_scores, average="weighted")

    avg_ms = float(np.mean(timings) * 1000.0) if timings else None
    
    return test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_f1_per_class, avg_ms, test_scores, test_auc_macro, test_auc_weighted, test_auprc_macro, test_auprc_weighted

def multilabel_auc_score(labels, scores, average="macro"):
    try:
        auc = roc_auc_score(labels, scores, average=average)
    except ValueError:
        # 過濾整個 array NaN
        labels = np.array(labels)
        scores = np.array(scores)
        
        if not np.isfinite(scores).all():
            mask = np.isfinite(scores)
            labels = labels[mask[:,0], :] if scores.ndim>1 else labels[mask]
            scores = scores[mask[:,0], :] if scores.ndim>1 else scores[mask]

        auc_scores = []
        weights = []

        for c in range(labels.shape[1]):
            y_c = labels[:, c]
            s_c = scores[:, c]

            # 過濾 NaN
            mask = np.isfinite(s_c)
            y_c = y_c[mask]
            s_c = s_c[mask]

            # 必須同時有正、負樣本
            if y_c.size == 0 or np.unique(y_c).size < 2:
                continue

            try:
                score = roc_auc_score(y_c, s_c)
            except ValueError:
                continue
            
            auc_scores.append(score)
            if average == "weighted":
                weights.append(y_c.sum())  # 正樣本數當權重

        if len(auc_scores) == 0:
            auc = np.nan
        if average == "weighted":
            if len(weights) == 0 or sum(weights) == 0:
                # 避免 ZeroDivisionError
                return np.mean(auc_scores)
            auc = np.average(auc_scores, weights=weights)
        auc = np.mean(auc_scores)

    return auc

def multilabel_auprc_score(labels, scores, average="macro"):
    try:
        auprc = average_precision_score(labels, scores, average=average)
    except ValueError:
        # 過濾整個 array NaN
        labels = np.array(labels)
        scores = np.array(scores)
        
        if not np.isfinite(scores).all():
            mask = np.isfinite(scores)
            labels = labels[mask[:,0], :] if scores.ndim>1 else labels[mask]
            scores = scores[mask[:,0], :] if scores.ndim>1 else scores[mask]

        auprc_scores = []
        weights = []

        for c in range(labels.shape[1]):
            y_c = labels[:, c]
            s_c = scores[:, c]

            # 過濾 NaN
            mask = np.isfinite(s_c)
            y_c = y_c[mask]
            s_c = s_c[mask]

            if y_c.size == 0 or np.unique(y_c).size < 2:
                continue

            try:
                score = average_precision_score(y_c, s_c)
            except ValueError:
                continue

            auprc_scores.append(score)
            if average == "weighted":
                weights.append(y_c.sum())

        if len(auprc_scores) == 0:
            auprc = np.nan
        if average == "weighted":
            if len(weights) == 0 or sum(weights) == 0:
                # 避免 ZeroDivisionError
                return np.mean(auprc_scores)
            auprc = np.average(auprc_scores, weights=weights)
        auprc = np.mean(auprc_scores)
    
    return auprc
