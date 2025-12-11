import argparse
import io
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import multilabel_confusion_matrix
from tqdm import tqdm

from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model
from utils.visualization_advanced import plot_confusion_matrix_heatmap, plot_true_confusion_matrix, plot_per_class_statistics


def compute_model_size_mb(model, checkpoint_path=None):
    path = checkpoint_path or getattr(model, "_checkpoint_path", None)
    if path and os.path.exists(path):
        size_bytes = os.path.getsize(path)
    else:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size_bytes = buffer.tell()
    return round(size_bytes / (1024 ** 2), 2)


def evaluate_test_set(
    data_path="data",
    checkpoint_path="checkpoints/lsnet_final.pth",
    batch_size=32,
    device=None,
    results_dir="results",
):
    device = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    os.makedirs(results_dir, exist_ok=True)

    # Load test dataset using lazy loading (same as main.py)
    test_ds = nih_chest_dataset(data_path=data_path, split="test", return_labels=False)
    class_names = test_ds.my_classes
    num_classes = len(class_names)
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    model = load_lsnet_model(
        num_classes=num_classes, device=device, checkpoint_path=checkpoint_path
    )
    model.eval()

    all_preds = []
    all_labels = []
    timings = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            start = time.perf_counter()
            logits = model(imgs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) / imgs.size(0))

            batch_preds = (torch.sigmoid(logits) > 0.5).cpu()
            all_preds.append(batch_preds)
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    accuracy = 1 - np.mean(np.abs(preds - labels))
    conf_matrix = multilabel_confusion_matrix(labels, preds)
    avg_inference_ms = float(np.mean(timings) * 1000.0) if timings else None
    model_size_mb = compute_model_size_mb(model, checkpoint_path)

    conf_records = []
    for idx, mat in enumerate(conf_matrix):
        tn, fp, fn, tp = mat.ravel()
        conf_records.append(
            {
                "class": class_names[idx],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    metrics = {
        "accuracy": float(accuracy),
        "avg_inference_ms_per_image": avg_inference_ms,
        "model_size_mb": model_size_mb,
        "num_samples": int(len(test_loader.dataset)),
        "checkpoint_path": checkpoint_path
        if checkpoint_path and os.path.exists(checkpoint_path)
        else getattr(model, "_checkpoint_path", None),
        "device": device,
    }

    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(results_dir, "confusion_matrix.json"), "w") as f:
        json.dump(conf_records, f, indent=2)

    np.save(os.path.join(results_dir, "confusion_matrix.npy"), conf_matrix)

    print(f"Saved metrics to {os.path.join(results_dir, 'test_metrics.json')}")
    print(f"Saved confusion matrix to {os.path.join(results_dir, 'confusion_matrix.json')}")
    
    # Plot true confusion matrix (14x14 actual vs predicted)
    print("\nGenerating true confusion matrix...")
    plot_true_confusion_matrix(labels, preds, class_names=class_names,
                               save_path=os.path.join(results_dir, 'true_confusion_matrix.png'))
    
    # Plot per-class metrics heatmap with disease names
    print("Generating per-class metrics heatmap...")
    plot_confusion_matrix_heatmap(conf_records, class_names=class_names, 
                                  save_path=os.path.join(results_dir, 'confusion_matrix_metrics.png'))
    
    # Save detailed per-class statistics
    print("Saving per-class statistics...")
    plot_per_class_statistics(conf_records, 
                            save_path=os.path.join(results_dir, 'per_class_statistics.json'))
    
    return metrics, conf_records


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSNet on NIH Chest X-ray test set")
    parser.add_argument("--data-path", default="data", help="Directory containing processed .pt files")
    parser.add_argument(
        "--checkpoint-path",
        default="checkpoints/lsnet_final.pth",
        help="Path to trained checkpoint; falls back to hub weights if missing",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Test batch size")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run on (cuda or cpu). Defaults to auto-detect",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write metrics and confusion matrix",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_test_set(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
        results_dir=args.results_dir,
    )
