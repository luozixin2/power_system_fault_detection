import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from typing import Dict, Any

class LSTMModel(BaseModel):
    """LSTM模型用于时序特征学习"""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, output_dim, config)
        
        self.hidden_size = config.get('lstm_hidden_size', 128)
        self.num_layers = config.get('lstm_num_layers', 2)
        self.dropout = config.get('lstm_dropout', 0.2)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层
        fc_layers = []
        prev_dim = self.hidden_size
        
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
    
    def _initialize_weights(self):
        """初始化权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置forget gate的偏置为1
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: shape (batch_size, seq_len, input_dim) 或 (batch_size, input_dim)
        """
        # 如果输入是2D，添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 全连接层
        output = self.fc(last_output)
        
        return output