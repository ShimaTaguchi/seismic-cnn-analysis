import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging
from typing import Dict, List, Optional

def plot_history(history: Dict[str, list], save_dir: str) -> None:
    """学習曲線のプロットと保存"""

    num_epochs_total = len(history['train_acc'])
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs_total + 1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, num_epochs_total + 1), history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs_total + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, num_epochs_total + 1), history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()

    seed_name = os.path.basename(save_dir)
    save_path = os.path.join(save_dir, f'learning_{seed_name}.png')
    plt.savefig(save_path)
    print(f"Learning curves plot: {save_path}")
    plt.close()

def evaluate_model(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        class_names: List[str], 
        save_dir: str, 
        device: torch.device, 
        logger: Optional[logging.Logger] = None
        ) -> None:
    """モデルの評価指標（分類レポート、混同行列、ROC）を計算・出力する"""
    
    def _print(msg: str):
        if logger: logger.info(msg)
        else: print(msg)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 分類レポート
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    _print("\nClassification Report:")
    _print(report)

    # 混同行列
    cm = confusion_matrix(all_labels, all_preds)
    _print(f"Confusion Matrix:\n{cm}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    seed_name = os.path.basename(save_dir)
    save_path = os.path.join(save_dir, f'confusion_matrix_{seed_name}.png')
    plt.savefig(save_path)
    _print(f"Confusion matrix plot: {save_path}")
    plt.close()

    # ROC AUC
    all_labels_arr = np.array(all_labels)
    all_probs_arr = np.array(all_probs)
    n_classes = len(class_names)
    y_test_bin = label_binarize(all_labels_arr, classes=range(n_classes))
    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs_arr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        _print(f"{class_names[i]}: AUC = {roc_auc[i]:.4f}")
        
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    roc_save_path = os.path.join(save_dir, f'roc_{seed_name}.png')
    plt.savefig(roc_save_path)
    _print(f"ROC Curve plot: {roc_save_path}")
    plt.close()