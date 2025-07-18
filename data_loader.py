import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from feature_extractor import UniversalFeatureExtractor
from config import data_config, FAULT_TYPES
from utils import create_sliding_windows
import logging

logger = logging.getLogger(__name__)

class PowerSystemDataset(Dataset):
    """电力系统故障检测数据集 - 多标签版本"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        # 多标签分类：标签是float类型的multi-hot编码
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataManager:
    """数据管理器：负责数据加载、预处理和划分"""
    
    def __init__(self, data_dir: str, config=None):
        self.data_dir = data_dir
        self.config = config or data_config
        self.feature_extractor = UniversalFeatureExtractor()
        self.is_fitted = False
        
    def load_all_data(self) -> List[pd.DataFrame]:
        """加载所有网络的 CSV 文件"""
        all_dfs = []
        for net_name in os.listdir(self.data_dir):
            net_dir = os.path.join(self.data_dir, net_name)
            if not os.path.isdir(net_dir):
                continue
            for fn in os.listdir(net_dir):
                if fn.endswith('.csv'):
                    path = os.path.join(net_dir, fn)
                    try:
                        df = pd.read_csv(path)
                        df['network'] = net_name
                        all_dfs.append(df)
                    except Exception as e:
                        logger.warning(f"无法加载 {path}: {e}")
        logger.info(f"总共加载 {len(all_dfs)} 个仿真文件")
        return all_dfs
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        提取特征并划分训练/验证/测试集（多标签），
        返回 X_train, X_val, X_test, y_train, y_val, y_test
        """
        # 1. 加载所有数据
        dfs = self.load_all_data()
        if not dfs:
            raise ValueError("没有找到任何数据文件！")

        # 2. 特征 & 多标签 提取
        logger.info("提取通用特征和多标签...")
        features_list, labels_list = self.feature_extractor.fit_transform(dfs, self.config.window_size)
        X = np.array(features_list, dtype=np.float32)           # (N_samples, N_features)
        y = np.array(labels_list, dtype=np.float32)             # (N_samples, N_fault_types)

        logger.info(f"特征矩阵维度: {X.shape}")
        logger.info(f"标签矩阵维度: {y.shape}")
        logger.info(f"每类故障正样本数量: {np.sum(y, axis=0)}")

        # 3. 划分数据集（多标签时不能用 stratify）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_ratio,
            random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.val_ratio / (1 - self.config.test_ratio),
            random_state=42
        )

        logger.info(f"训练集样本数: {X_train.shape[0]}")
        logger.info(f"验证集样本数: {X_val.shape[0]}")
        logger.info(f"测试集样本数: {X_test.shape[0]}")

        self.is_fitted = True
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_windowed_data(self, dataframes: List[pd.DataFrame], 
                           labels: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """创建滑动窗口数据"""
        windowed_features = []
        windowed_labels = []
        
        for df, df_labels in zip(dataframes, labels):
            # 创建特征窗口
            feature_windows = create_sliding_windows(
                df.select_dtypes(include=[np.number]).values,
                self.config.window_size,
                self.config.overlap_ratio
            )
            
            # 创建标签窗口
            label_windows = create_sliding_windows(
                df_labels,
                self.config.window_size,
                self.config.overlap_ratio
            )
            
            # 使用窗口最后一个时间步的标签
            window_labels = label_windows[:, -1]
            
            windowed_features.append(feature_windows)
            windowed_labels.append(window_labels)
        
        return windowed_features, windowed_labels
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        提取特征并划分训练/验证/测试集（多标签），
        返回 X_train, X_val, X_test, y_train, y_val, y_test
        """
        # 1. 加载所有数据
        dfs = self.load_all_data()
        if not dfs:
            raise ValueError("没有找到任何数据文件！")

        # 2. 特征 & 多标签 提取
        logger.info("提取通用特征和多标签...")
        features_list, labels_list = self.feature_extractor.fit_transform(dfs, self.config.window_size)
        X = np.array(features_list, dtype=np.float32)           # (N_samples, N_features)
        y = np.array(labels_list, dtype=np.float32)             # (N_samples, N_fault_types)

        logger.info(f"特征矩阵维度: {X.shape}")
        logger.info(f"标签矩阵维度: {y.shape}")
        logger.info(f"每类故障正样本数量: {np.sum(y, axis=0)}")

        # 3. 划分数据集（多标签时不能用 stratify）
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_ratio,
            random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.val_ratio / (1 - self.config.test_ratio),
            random_state=42
        )

        logger.info(f"训练集样本数: {X_train.shape[0]}")
        logger.info(f"验证集样本数: {X_val.shape[0]}")
        logger.info(f"测试集样本数: {X_test.shape[0]}")

        self.is_fitted = True
        return X_train, X_val, X_test, y_train, y_val, y_test
    

    def create_data_loaders(self,
                            X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                            batch_size: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """根据划分好的数据创建 PyTorch DataLoader"""
        batch_size = batch_size or self.config.batch_size

        # ==== 1) 根据 config.sampling_strategy 对 X_train/y_train 重新采样 ====
        strat = self.config.sampling_strategy.lower()
        if strat != 'none':
            idxs = np.arange(len(y_train))
            is_normal = (y_train.sum(axis=1) == 0)
            normal_idxs = idxs[is_normal]
            fault_idxs  = idxs[~is_normal]

            if strat == 'undersample':
                keep_n = int(len(normal_idxs) * self.config.undersample_negative_ratio)
                keep_n = max(1, keep_n)
                sel_normal = np.random.choice(normal_idxs, keep_n, replace=False)
                sel = np.concatenate([sel_normal, fault_idxs])

            elif strat == 'oversample':
                desired_pos = int(len(normal_idxs) * self.config.oversample_positive_ratio)
                extra = max(0, desired_pos - len(fault_idxs))
                if extra > 0:
                    dup = np.random.choice(fault_idxs, extra, replace=True)
                    X_train = np.vstack([X_train, X_train[dup]])
                    y_train = np.vstack([y_train, y_train[dup]])
                sel = np.arange(len(y_train))
            elif strat == 'smote':
                from imblearn.over_sampling import SMOTE
                # 1) 计算各类正样本数，取最大值
                pos_counts = y_train.sum(axis=0).astype(int)       # shape (n_labels,)
                max_pos = int(pos_counts.max())
                X_syn_list, y_syn_list = [], []

                for cls_idx, cnt in enumerate(pos_counts):
                    need = max_pos - cnt
                    if need <= 0 or cnt < 2:
                        continue  # 已经够或样本太少跳过
                    # 构造二分类标签（0/1）
                    y_bin = (y_train[:, cls_idx] >= 0.5).astype(int)
                    # SMOTE 需要完整的训练集和二分类标签
                    sm = SMOTE(
                        sampling_strategy={1: max_pos},
                        k_neighbors=min(self.config.smote_k_neighbors, cnt-1),
                        random_state=42
                    )
                    X_res, y_res = sm.fit_resample(X_train, y_bin)
                    # 末尾 synthetic_count = max_pos - cnt
                    synthetic_count = max_pos - cnt
                    # 取最后 synthetic_count 条 label==1 的行
                    # SMOTE 会将合成样本追加在后面
                    X_new = X_res[-synthetic_count:]
                    # 构建多标签标记：该 cls_idx 为 1，其它保持 0
                    y_new = np.zeros((synthetic_count, y_train.shape[1]), dtype=float)
                    y_new[:, cls_idx] = 1.0

                    X_syn_list.append(X_new)
                    y_syn_list.append(y_new)

                if X_syn_list:
                    X_aug = np.vstack([X_train] + X_syn_list)
                    y_aug = np.vstack([y_train] + y_syn_list)
                    # 打乱顺序
                    perm = np.random.permutation(len(X_aug))
                    X_train, y_train = X_aug[perm], y_aug[perm]
                    logger.info(f"SMOTE 合成后，训练集从 {len(y_train)-sum(len(a) for a in y_syn_list)} ➔ {len(y_train)}")

            else:
                sel = np.arange(len(y_train))
            # shuffle
            if strat != 'smote':
                perm = np.random.permutation(len(sel))
                sel = sel[perm]
                X_train, y_train = X_train[sel], y_train[sel]

                logger.info(f"After '{strat}' sampling: train size = {len(y_train)}")

        # ==== 2) 构造 DataLoader ====
        train_ds = PowerSystemDataset(X_train, y_train)
        val_ds   = PowerSystemDataset(X_val,   y_val)
        test_ds  = PowerSystemDataset(X_test,  y_test)

        loader_args = dict(batch_size=batch_size,
                           num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True,  **loader_args)
        val_loader   = DataLoader(val_ds,   shuffle=False, **loader_args)
        test_loader  = DataLoader(test_ds,  shuffle=False, **loader_args)

        return train_loader, val_loader, test_loader
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        if not self.is_fitted:
            raise ValueError("DataManager must be fitted first")
        return self.feature_extractor.scaler.n_features_in_
    
    def analyze_data_distribution(self, labels: np.ndarray):
        """分析数据分布"""
        logger.info("Data Distribution Analysis:")
        for i, fault_type in enumerate(FAULT_TYPES):
            count = np.sum(labels == i)
            percentage = count / len(labels) * 100
            logger.info(f"{fault_type}: {count} samples ({percentage:.2f}%)")