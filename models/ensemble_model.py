import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .lstm_model import LSTMModel
from .cnn_model import CNNModel
from typing import Dict, Any

class EnsembleModel(BaseModel):
    """集成模型：结合LSTM和CNN"""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, output_dim, config)
        
        # 创建子模型
        self.lstm_model = LSTMModel(input_dim, output_dim, config)
        self.cnn_model = CNNModel(input_dim, output_dim, config)
        
        # 融合层
        fusion_dim = config.get('fusion_dim', 128)
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim) if config.get('batch_norm', True) else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(fusion_dim, output_dim)
        )
        
        # 注意力机制权重
        self.attention = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        self.ensemble_method = config.get('ensemble_method', 'attention')  # 'concat', 'average', 'attention'
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in [self.fusion_layer, self.attention]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 获取各子模型的输出
        lstm_output = self.lstm_model(x)
        cnn_output = self.cnn_model(x)
        
        if self.ensemble_method == 'average':
            # 简单平均
            output = (lstm_output + cnn_output) / 2
        
        elif self.ensemble_method == 'concat':
            # 拼接后通过融合层
            combined = torch.cat([lstm_output, cnn_output], dim=1)
            output = self.fusion_layer(combined)
        
        elif self.ensemble_method == 'attention':
            # 注意力机制加权
            combined = torch.cat([lstm_output, cnn_output], dim=1)
            attention_weights = self.attention(combined)  # (batch_size, 2)
            
            # 计算加权输出
            weighted_lstm = lstm_output * attention_weights[:, 0:1]
            weighted_cnn = cnn_output * attention_weights[:, 1:2]
            output = weighted_lstm + weighted_cnn
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力权重"""
        if self.ensemble_method != 'attention':
            return None
        
        with torch.no_grad():
            lstm_output = self.lstm_model(x)
            cnn_output = self.cnn_model(x)
            combined = torch.cat([lstm_output, cnn_output], dim=1)
            attention_weights = self.attention(combined)
        
        return attention_weights