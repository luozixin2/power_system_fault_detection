import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
import os

from models import LSTMModel, CNNModel, EnsembleModel, TransformerModel
from config import model_config, training_config, FAULT_TYPES
from utils import AsymmetricFocalLoss, EarlyStopping, FocalLoss, calculate_class_weights, save_model_checkpoint, get_device

logger = logging.getLogger(__name__)

class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or training_config
        
        # 设置设备
        self.device = get_device(self.config.device)
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 设置学习率调度器
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        else:
            self.scheduler = None
        
        # 设置损失函数
        if self.config.use_focal_loss:
            if self.config.use_asymmetric_focal_loss:
                self.criterion = AsymmetricFocalLoss(
                    alpha=self.config.asymmetric_focal_alpha,
                    gamma_pos=self.config.asymmetric_focal_gamma_pos,
                    gamma_neg=self.config.asymmetric_focal_gamma_neg
                )
            else:
                self.criterion = FocalLoss(alpha=self.config.focal_alpha,
                                       gamma=self.config.focal_gamma)
        elif self.config.use_class_weights:
            # 计算正样本权重
            all_labels = []
            for _, labels in train_loader:
                all_labels.append(labels.numpy())
            all_labels = np.vstack(all_labels)
            
            # 计算每个类别的正样本比例
            pos_weight = []
            for i in range(len(FAULT_TYPES)):
                pos_count = np.sum(all_labels[:, i])
                neg_count = len(all_labels) - pos_count
                if pos_count > 0:
                    weight = neg_count / pos_count
                else:
                    weight = 1.0
                pos_weight.append(weight)
            
            pos_weight = torch.FloatTensor(pos_weight).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计 - 多标签分类的准确率计算
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5  # 阈值0.5
            correct += (predictions == targets.bool()).float().mean().item()
            total += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validating"):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                correct += (predictions == targets.bool()).float().mean().item()
                total += 1
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, save_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """训练模型"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        logger.info(f"Model size: {self.model.get_model_size():.2f} MB")
        
        for epoch in range(self.config.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.config.epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                save_model_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, best_model_path
                )
                logger.info(f"New best model saved: {best_model_path}")
            
            # 早停检查
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed!")
        return self.history

def create_model(model_type: str, input_dim: int, output_dim: int, config=None) -> nn.Module:
    """创建模型"""
    config = config or model_config.__dict__
    
    if model_type == 'lstm':
        return LSTMModel(input_dim, output_dim, config)
    elif model_type == 'cnn':
        return CNNModel(input_dim, output_dim, config)
    elif model_type == 'ensemble':
        return EnsembleModel(input_dim, output_dim, config)
    elif model_type == 'transformer':
        return TransformerModel(input_dim, output_dim, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(train_loader: DataLoader, val_loader: DataLoader, 
                input_dim: int, model_type: str = 'ensemble',
                save_dir: str = 'checkpoints') -> Tuple[nn.Module, Dict[str, List[float]]]:
    """训练模型的主函数"""
    # 创建模型
    model = create_model(model_type, input_dim, len(FAULT_TYPES))
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader)
    
    # 训练
    history = trainer.train(save_dir)
    
    return model, history