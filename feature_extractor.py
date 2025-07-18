import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.fft import fft, fftfreq
import pywt
from sklearn.preprocessing import StandardScaler
import networkx as nx
import warnings
from config import feature_config, FAULT_TYPES

class UniversalFeatureExtractor:
    """通用特征提取器 - 将不同网络的变长特征转换为固定维度特征"""
    
    def __init__(self, config=None):
        self.config = config or feature_config
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_statistical_features(self, values: np.ndarray) -> Dict[str, float]:
        """提取统计特征（修复版本）"""
        if len(values) == 0:
            return {feat: 0.0 for feat in self.config.statistical_features}
        
        features = {}
        
        # 检查数据变异性，避免数值计算问题
        data_std = np.std(values)
        data_range = np.ptp(values)  # peak to peak
        is_nearly_constant = data_std < 1e-10 or data_range < 1e-10
        
        if 'mean' in self.config.statistical_features:
            features['mean'] = np.mean(values)
        if 'std' in self.config.statistical_features:
            features['std'] = data_std
        if 'min' in self.config.statistical_features:
            features['min'] = np.min(values)
        if 'max' in self.config.statistical_features:
            features['max'] = np.max(values)
        if 'median' in self.config.statistical_features:
            features['median'] = np.median(values)
        
        # 安全计算偏度，避免精度损失警告
        if 'skewness' in self.config.statistical_features:
            if is_nearly_constant or len(values) < 3:
                # 数据几乎恒定或样本太少时，偏度为0
                features['skewness'] = 0.0
            else:
                try:
                    # 使用警告过滤器忽略精度损失警告
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        skew_val = stats.skew(values)
                        # 检查结果是否有限
                        features['skewness'] = skew_val if np.isfinite(skew_val) else 0.0
                except (ValueError, ZeroDivisionError):
                    features['skewness'] = 0.0
        
        # 安全计算峰度，避免精度损失警告
        if 'kurtosis' in self.config.statistical_features:
            if is_nearly_constant or len(values) < 4:
                # 数据几乎恒定或样本太少时，峰度为0（正态分布的超峰度）
                features['kurtosis'] = 0.0
            else:
                try:
                    # 使用警告过滤器忽略精度损失警告
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        kurt_val = stats.kurtosis(values)
                        # 检查结果是否有限
                        features['kurtosis'] = kurt_val if np.isfinite(kurt_val) else 0.0
                except (ValueError, ZeroDivisionError):
                    features['kurtosis'] = 0.0
                    
        if 'rms' in self.config.statistical_features:
            features['rms'] = np.sqrt(np.mean(values**2))
        if 'peak_to_peak' in self.config.statistical_features:
            features['peak_to_peak'] = data_range
            
        return features
    
    def extract_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        """提取频谱特征（改进版本）"""
        features = {}
        
        if len(signal) == 0:
            return features
        
        # FFT特征
        if self.config.use_spectral_features:
            # 确保信号长度足够
            if len(signal) < self.config.fft_size:
                signal = np.pad(signal, (0, self.config.fft_size - len(signal)), 'constant')
            elif len(signal) > self.config.fft_size:
                signal = signal[:self.config.fft_size]
            
            # 检查信号是否几乎恒定
            if np.std(signal) < 1e-10:
                # 对于恒定信号，设置默认频谱特征
                features['spectral_centroid'] = 0.0
                features['spectral_bandwidth'] = 0.0
                features['spectral_rolloff'] = 0.0
                features['spectral_flux'] = 0.0
            else:
                fft_vals = np.abs(fft(signal)[:self.config.fft_size//2])
                freqs = fftfreq(self.config.fft_size)[:self.config.fft_size//2]
                
                # 避免除零错误
                fft_sum = np.sum(fft_vals)
                if fft_sum > 1e-10:
                    features['spectral_centroid'] = np.sum(freqs * fft_vals) / fft_sum
                    centroid = features['spectral_centroid']
                    features['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - centroid)**2) * fft_vals) / fft_sum)
                    
                    # 频谱滚降点
                    cumsum_fft = np.cumsum(fft_vals)
                    rolloff_threshold = 0.85 * fft_sum
                    rolloff_idx = np.where(cumsum_fft >= rolloff_threshold)[0]
                    features['spectral_rolloff'] = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
                else:
                    features['spectral_centroid'] = 0.0
                    features['spectral_bandwidth'] = 0.0
                    features['spectral_rolloff'] = 0.0
                
                # 频谱通量
                features['spectral_flux'] = np.sum(np.diff(fft_vals)**2) if len(fft_vals) > 1 else 0.0
            
            # 小波特征
            if len(signal) >= 2**self.config.wavelet_levels:
                try:
                    coeffs = pywt.wavedec(signal, self.config.wavelet_name, level=self.config.wavelet_levels)
                    for i, coeff in enumerate(coeffs):
                        if len(coeff) > 0:
                            features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
                            features[f'wavelet_std_level_{i}'] = np.std(coeff)
                        else:
                            features[f'wavelet_energy_level_{i}'] = 0.0
                            features[f'wavelet_std_level_{i}'] = 0.0
                except Exception:
                    # 小波变换失败时设置默认值
                    for i in range(self.config.wavelet_levels + 1):
                        features[f'wavelet_energy_level_{i}'] = 0.0
                        features[f'wavelet_std_level_{i}'] = 0.0
        
        return features
    
    def extract_temporal_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """提取时序特征（改进版本）"""
        features = {}
        
        if len(time_series) < 2:
            return features
        
        # 检查时间序列变异性
        series_std = np.std(time_series)
        is_nearly_constant = series_std < 1e-10
        
        # 差分特征
        diff1 = np.diff(time_series)
        features['diff1_mean'] = np.mean(diff1)
        features['diff1_std'] = np.std(diff1)
        
        if len(diff1) > 1:
            diff2 = np.diff(diff1)
            features['diff2_mean'] = np.mean(diff2)
            features['diff2_std'] = np.std(diff2)
        
        # 趋势特征
        if len(time_series) > 1 and not is_nearly_constant:
            try:
                x = np.arange(len(time_series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, time_series)
                features['trend_slope'] = slope if np.isfinite(slope) else 0.0
                features['trend_r_squared'] = r_value**2 if np.isfinite(r_value) else 0.0
            except Exception:
                features['trend_slope'] = 0.0
                features['trend_r_squared'] = 0.0
        else:
            features['trend_slope'] = 0.0
            features['trend_r_squared'] = 0.0
        
        # 变化率特征
        if len(time_series) > 1:
            changes = np.abs(np.diff(time_series))
            features['change_rate_mean'] = np.mean(changes)
            features['change_rate_max'] = np.max(changes) if len(changes) > 0 else 0.0
            
            # 峰值检测（仅在信号有变化时进行）
            if not is_nearly_constant:
                features['num_peaks'] = len(self._find_peaks(time_series))
            else:
                features['num_peaks'] = 0
        
        return features
    
    def _find_peaks(self, signal: np.ndarray, prominence: float = None) -> List[int]:
        """改进的峰值检测"""
        if len(signal) < 3:
            return []
        
        # 自适应prominence阈值
        if prominence is None:
            signal_std = np.std(signal)
            prominence = max(signal_std * 0.1, 1e-6)
        
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] - min(signal[i-1], signal[i+1]) > prominence:
                    peaks.append(i)
        return peaks
    
    def extract_topology_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """提取拓扑特征"""
        features = {}
        
        # 统计各类设备数量
        features['num_buses'] = len([col for col in df.columns if col.startswith('bus_') and col.endswith('_vm_pu')])
        features['num_lines'] = len([col for col in df.columns if col.startswith('line_') and col.endswith('_loading_percent')])
        features['num_generators'] = len([col for col in df.columns if col.startswith('gen_') and col.endswith('_p_mw')])
        features['num_loads'] = len([col for col in df.columns if col.startswith('load_') and col.endswith('_p_mw')])
        
        # 归一化设备数量特征
        total_elements = features['num_buses'] + features['num_lines'] + features['num_generators'] + features['num_loads']
        if total_elements > 0:
            features['bus_ratio'] = features['num_buses'] / total_elements
            features['line_ratio'] = features['num_lines'] / total_elements
            features['gen_ratio'] = features['num_generators'] / total_elements
            features['load_ratio'] = features['num_loads'] / total_elements
        else:
            features['bus_ratio'] = 0.0
            features['line_ratio'] = 0.0
            features['gen_ratio'] = 0.0
            features['load_ratio'] = 0.0
        
        return features
    
    def extract_system_level_features(self, df: pd.DataFrame, time_window: np.ndarray) -> Dict[str, float]:
        """提取系统级特征（改进版本）"""
        features = {}
        
        # 从系统级列提取特征
        system_columns = ['total_loss_mw', 'total_generation_mw', 'total_load_mw']
        
        for col in system_columns:
            if col in df.columns:
                values = df[col].iloc[time_window].values
                # 过滤非数值和无穷值
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    col_features = self.extract_statistical_features(values)
                    for feat_name, feat_val in col_features.items():
                        features[f'{col}_{feat_name}'] = feat_val
                    
                    # 时序特征
                    if self.config.use_temporal_features and len(values) > 1:
                        temporal_feats = self.extract_temporal_features(values)
                        for feat_name, feat_val in temporal_feats.items():
                            features[f'{col}_temporal_{feat_name}'] = feat_val
        
        # 电压特征聚合
        voltage_cols = [col for col in df.columns if '_vm_pu' in col]
        if voltage_cols:
            voltage_data = df[voltage_cols].iloc[time_window].values.flatten()
            voltage_data = voltage_data[np.isfinite(voltage_data)]
            if len(voltage_data) > 0:
                voltage_features = self.extract_statistical_features(voltage_data)
                for feat_name, feat_val in voltage_features.items():
                    features[f'voltage_{feat_name}'] = feat_val
        
        # 线路负载率特征聚合
        loading_cols = [col for col in df.columns if '_loading_percent' in col]
        if loading_cols:
            loading_data = df[loading_cols].iloc[time_window].values.flatten()
            loading_data = loading_data[np.isfinite(loading_data)]
            if len(loading_data) > 0:
                loading_features = self.extract_statistical_features(loading_data)
                for feat_name, feat_val in loading_features.items():
                    features[f'loading_{feat_name}'] = feat_val
        
        # 发电机功率特征聚合
        gen_p_cols = [col for col in df.columns if col.startswith('gen_') and col.endswith('_p_mw')]
        if gen_p_cols:
            gen_p_data = df[gen_p_cols].iloc[time_window].values.flatten()
            gen_p_data = gen_p_data[np.isfinite(gen_p_data)]
            if len(gen_p_data) > 0:
                gen_features = self.extract_statistical_features(gen_p_data)
                for feat_name, feat_val in gen_features.items():
                    features[f'generation_{feat_name}'] = feat_val
        
        return features
    
    def extract_window_features(self, df: pd.DataFrame, time_window: np.ndarray) -> Dict[str, float]:
        """提取时间窗口的所有特征"""
        all_features = {}
        
        # 系统级特征
        system_features = self.extract_system_level_features(df, time_window)
        all_features.update(system_features)
        
        # 拓扑特征（只需要计算一次）
        if self.config.use_topology_features:
            topology_features = self.extract_topology_features(df)
            all_features.update(topology_features)
        
        # 收敛性特征
        if 'converged' in df.columns:
            converged_values = df['converged'].iloc[time_window].values
            all_features['convergence_rate'] = np.mean(converged_values)
            all_features['num_convergence_failures'] = np.sum(~converged_values)
        
        # 故障状态特征
        for fault_type in FAULT_TYPES:
            if fault_type in df.columns:
                fault_values = df[fault_type].iloc[time_window].values
                all_features[f'{fault_type}_active_ratio'] = np.mean(fault_values)
        
        return all_features
    
    def fit_transform(self, dataframes: List[pd.DataFrame], window_size: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """拟合并转换数据"""
        all_feature_dicts: List[Dict] = []
        all_labels: List[np.ndarray] = []

        # 提取所有特征（dict 形式）
        for df in dataframes:
            feature_dicts, labels = self._extract_features_from_dataframe(df, window_size)
            all_feature_dicts.extend(feature_dicts)
            all_labels.extend(labels)

        # 用 DataFrame 对齐所有 feature 字典，缺失项自动填 NaN
        feature_df = pd.DataFrame(all_feature_dicts)
        # 如果你希望把 NaN 填 0，可以：
        feature_df = feature_df.fillna(0.0)

        feature_array = feature_df.values.astype(float)
        # 拟合标准化器
        self.scaler.fit(feature_array)
        self.is_fitted = True
        # 标准化
        normalized_features = self.scaler.transform(feature_array)

        return normalized_features, all_labels
    
    def transform(self, dataframes: List[pd.DataFrame], window_size: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """转换数据（已拟合）"""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        all_feature_dicts: List[Dict] = []
        all_labels: List[np.ndarray] = []
        
        # 提取所有特征
        for df in dataframes:
            feature_dicts, labels = self._extract_features_from_dataframe(df, window_size)
            all_feature_dicts.extend(feature_dicts)
            all_labels.extend(labels)

        feature_df = pd.DataFrame(all_feature_dicts).fillna(0.0)
        feature_array = feature_df.values.astype(float)
        normalized_features = self.scaler.transform(feature_array)
        return normalized_features, all_labels
    
    def _extract_features_from_dataframe(self, df: pd.DataFrame, window_size: int) -> Tuple[List[Dict], List[np.ndarray]]:
        """从单个数据框提取特征"""
        features_list = []
        labels_list = []
        
        # 创建滑动窗口
        for start_idx in range(0, len(df) - window_size + 1, window_size // 2):  # 50%重叠
            end_idx = start_idx + window_size
            time_window = np.arange(start_idx, end_idx)
            
            # 提取特征
            window_features = self.extract_window_features(df, time_window)
            features_list.append(list(window_features.values()))
            
            # 提取多标签（使用窗口中的多数投票或最后时间步）
            window_labels = []
            for fault_type in FAULT_TYPES:
                if fault_type in df.columns:
                    # 使用窗口中的多数投票来决定标签
                    fault_values = df[fault_type].iloc[time_window].values
                    majority_label = 1 if np.mean(fault_values) > 0.5 else 0
                    window_labels.append(majority_label)
                else:
                    window_labels.append(0)
            labels_list.append(np.array(window_labels, dtype=np.float32))
        
        return features_list, labels_list