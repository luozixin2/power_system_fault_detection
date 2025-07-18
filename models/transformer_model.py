import math
import torch
import torch.nn as nn
from .base_model import BaseModel
from typing import Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerModel(BaseModel):
    """基于 Transformer Encoder 的序列分类/多标签模型"""

    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__(input_dim, output_dim, config)
        d_model = config.get('transformer_d_model', 128)
        nhead = config.get('transformer_nhead', 4)
        num_layers = config.get('transformer_num_layers', 3)
        dim_feedforward = config.get('transformer_dim_feedforward', 256)
        dropout = config.get('transformer_dropout', 0.1)
        max_len = config.get('transformer_max_len', 500)

        # 输入 embedding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 分类 head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, config.get('hidden_dims', [256])[0]),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(config.get('hidden_dims', [256])[0], output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        # 简单初始化
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) 或 (batch, input_dim)
        """
        # 如果输入是 (batch, input_dim)，当作 seq_len=1
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)

        # 投影到 d_model
        x = self.input_proj(x)           # (batch, seq_len, d_model)
        x = self.pos_encoder(x)          # 加上位置编码
        x = x.transpose(0, 1)            # Transformer 要求 (seq_len, batch, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        x = x.transpose(0, 1)            # (batch, seq_len, d_model)

        # 池化：取序列第一 token 或者 mean pooling
        # 这里用 mean pooling
        x = x.mean(dim=1)                # (batch, d_model)

        # 分类
        out = self.classifier(x)         # (batch, output_dim)
        return out