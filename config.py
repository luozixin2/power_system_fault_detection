import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

# 故障类型映射
FAULT_TYPES = [
    # "normal", # 正常状态
    "single_line_outage",
    "double_line_outage",
    "generator_outage", 
    "load_spike_moderate",
    "load_spike_severe",
    "generator_instability",
    "line_impedance_drift",    # 渐变性故障
    "insulation_degradation",  # 渐变性故障
    "protection_malfunction",  # 保护误动作
    "voltage_regulator_fault",
    "cascading_failure",       # 连锁故障
    "intermittent_fault",      # 间歇性故障
]

@dataclass
class DataConfig:
    """数据相关配置"""
    dataset_dir: str = "dynamic_simulation_datasets"
    window_size: int = 30  # 时间窗口大小
    overlap_ratio: float = 0.5  # 窗口重叠比例
    test_ratio: float = 0.2
    val_ratio: float = 0.2
    # —— 新增采样策略 —— 
    # none / undersample / oversample / smote
    sampling_strategy: str = "none"
    # undersample 时保留负样本比例
    undersample_negative_ratio: float = 1.0
    # oversample 时正样本倍率
    oversample_positive_ratio: float = 1.0
    # smote 时每类合成后正样本数 = max_class_count
    smote_k_neighbors: int = 5  # SMOTE 最近邻参数
    
@dataclass
class FeatureConfig:
    """特征提取配置"""
    # 统计特征
    use_statistical_features: bool = True
    statistical_features: List[str] = None
    
    # 频谱特征
    use_spectral_features: bool = True
    fft_size: int = 64
    wavelet_name: str = "db4"
    wavelet_levels: int = 3
    
    # 拓扑特征
    use_topology_features: bool = True
    
    # 时序特征
    use_temporal_features: bool = True
    
    def __post_init__(self):
        if self.statistical_features is None:
            self.statistical_features = [
                'mean', 'std', 'min', 'max', 'median', 
                'skewness', 'kurtosis', 'rms', 'peak_to_peak'
            ]

@dataclass
class ModelConfig:
    """模型配置"""
    model_type: str = "ensemble"  # lstm, cnn, ensemble
    
    # LSTM配置
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    
    # CNN配置
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None
    cnn_dropout: float = 0.2
    
    # Transformer配置
    transformer_d_model = 128
    transformer_nhead = 4
    transformer_num_layers = 3
    transformer_dim_feedforward = 256
    transformer_dropout = 0.1
    transformer_max_len = 500
    
    # 通用配置
    hidden_dims: List[int] = None
    output_dim: int = len(FAULT_TYPES)  # 故障类型数量
    dropout: float = 0.3
    batch_norm: bool = True
    
    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]

@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 15
    min_delta: float = 0.001
    
    # 学习率调度
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10
    
    # 类别权重
    use_class_weights: bool = False
    # 是否使用 Focal Loss
    use_focal_loss: bool = True
    # 仅在使用 Focal Loss 时生效
    use_asymmetric_focal_loss: bool = True
    # Focal Loss 参数
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Asymmetric Focal Loss 参数
    asymmetric_focal_alpha: float = 0.75
    asymmetric_focal_gamma_pos: float = 1.0
    asymmetric_focal_gamma_neg: float = 2.0 
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda
    


    # ===== 新增：多标签阈值配置 =====
    thresholds: List[float] = field(default_factory=lambda: [0.5]*len(FAULT_TYPES))

# 全局配置实例
data_config = DataConfig()
feature_config = FeatureConfig()
model_config = ModelConfig()
training_config = TrainingConfig()


# FAULT_TYPE_TO_ID = {fault: idx for idx, fault in enumerate(FAULT_TYPES)}