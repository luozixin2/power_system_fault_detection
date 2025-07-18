import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging

def set_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_logging(log_file: str = "training.log"):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_device(device_str: str = "auto") -> torch.device:
    """获取计算设备"""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    return device

def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """计算类别权重"""
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    return torch.FloatTensor(class_weights)

def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str = None):
    """绘制混淆矩阵"""
    # 强制使用所有标签，即使某些类别在 y_true/y_pred 中都没出现
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: List[str]):
    """打印分类报告，显式指定 labels，避免 label 数量不匹配报错"""
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0  # 如果某些类别没有样本，则记为 0
    )
    print("Classification Report:")
    print(report)
    return report

def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, loss: float, save_path: str):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)

def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                         checkpoint_path: str) -> Tuple[int, float]:
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 15, min_delta: float = 0.001, 
                 restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True
            return False

def create_sliding_windows(data: np.ndarray, window_size: int, 
                          overlap_ratio: float = 0.5) -> np.ndarray:
    """创建滑动窗口"""
    step_size = max(1, int(window_size * (1 - overlap_ratio)))
    windows = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)

class FocalLoss(nn.Module):
    """针对多标签二分类的 Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets 形状均为 (batch, num_classes)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probas = torch.sigmoid(logits)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss for multi-label binary classification,
       偏向提升少数类（故障）的召回率。
    Args:
      alpha: 正类 (target=1) 的权重，默认 0.75（>0.5 偏向正类）
      gamma_pos: 正类的聚焦系数 γ_pos，建议 0.5~1.0
      gamma_neg: 负类的聚焦系数 γ_neg，建议 2.0~4.0
      reduction: 'none' | 'mean' | 'sum'
    """
    def __init__(
        self,
        alpha: float = 0.75,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:   (batch, num_classes)
        targets:  (batch, num_classes), 0/1 多标签
        """
        # 1) 官方 BCE-with-logits
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # 2) 预测概率
        probas = torch.sigmoid(logits)
        # 3) p_t = p if y=1 else (1-p)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        # 4) α_t 根据正负样本分配权重
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        # 5) γ_t 根据正负样本选用不同的 γ
        gamma_t = targets * self.gamma_pos + (1 - targets) * self.gamma_neg
        # 6) 最终 loss
        loss = alpha_t * (1 - p_t) ** gamma_t * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss