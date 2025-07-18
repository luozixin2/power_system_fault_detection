import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from typing import Dict, Any, List

class CNNModel(BaseModel):
    """CNN模型用于频谱特征学习"""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, output_dim, config)
        
        self.filters = config.get('cnn_filters', [32, 64, 128])
        self.kernel_sizes = config.get('cnn_kernel_sizes', [3, 3, 3])
        self.dropout = config.get('cnn_dropout', 0.2)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        
        # 1D CNN层
        conv_layers = []
        in_channels = 1
        
        for filters, kernel_size in zip(self.filters, self.kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters) if config.get('batch_norm', True) else nn.Identity(),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(self.dropout)
            ])
            in_channels = filters
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 计算conv层输出维度
        self.conv_output_dim = self._get_conv_output_dim(input_dim)
        
        # 全连接层
        fc_layers = []
        prev_dim = self.conv_output_dim
        
        for hidden_dim in self.hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config.get('batch_norm', True) else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(config.get('dropout', 0.3))
            ])
            prev_dim = hidden_dim
        
        fc_layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _get_conv_output_dim(self, input_dim: int) -> int:
        """计算卷积层输出维度"""
        x = torch.randn(1, 1, input_dim)
        x = self.conv_layers(x)
        return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: shape (batch_size, input_dim)
        """
        # 添加channel维度
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # CNN前向传播
        x = self.conv_layers(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        output = self.fc(x)
        
        return output