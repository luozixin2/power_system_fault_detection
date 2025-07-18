import pandapower as pp
import pandapower.networks as nw
import numpy as np
import pandas as pd
import os
import copy
from tqdm import tqdm
import warnings
from scipy import stats
from datetime import datetime, timedelta

# --- 1. 增强的全局配置 ---
NETWORKS_TO_SIMULATE = {
    "case9": nw.case9,
    "case14": nw.case14,
    "case30": nw.case30,
    "case39": nw.case39, 
    "case57": nw.case57,
    "case118": nw.case118,
    "case145": nw.case145,
    "case300": nw.case300,
}

# 扩展的故障类型定义
FAULT_TYPES = [
    "normal",
    "single_line_outage",
    "double_line_outage", 
    "generator_outage",
    "transformer_outage",
    "bus_fault",
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

# 增强的仿真参数
SIMULATION_TIME_STEPS = 100    # 时间步数
SAMPLES_PER_CASE = 50         # 样本数
OUTPUT_DIR = "dynamic_simulation_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 新增配置
class SimulationConfig:
    # 噪声配置
    MEASUREMENT_NOISE_STD = 0.02      # 测量噪声标准差 (2%)
    VOLTAGE_NOISE_STD = 0.01          # 电压测量噪声 (1%)
    POWER_NOISE_STD = 0.03            # 功率测量噪声 (3%)
    
    # 负荷模式配置
    DAILY_LOAD_VARIATION = True       # 启用日负荷变化
    SEASONAL_VARIATION = True         # 启用季节性变化
    STOCHASTIC_LOAD_COMPONENT = True  # 启用随机负荷分量
    
    # 发电机配置
    GENERATOR_RAMPING = True          # 启用发电机爬坡
    RENEWABLE_GENERATION = True       # 启用可再生能源波动
    
    # 故障配置
    CASCADING_PROBABILITY = 0.15      # 连锁故障概率
    INTERMITTENT_FAULT_PROBABILITY = 0.1  # 间歇性故障概率
    DEGRADATION_RATE = 0.001          # 设备劣化率
    
    # 运行模式
    MAINTENANCE_MODE_PROBABILITY = 0.05   # 维护模式概率
    EMERGENCY_MODE_PROBABILITY = 0.02     # 应急模式概率
    
    # 数据质量
    MISSING_DATA_PROBABILITY = 0.01   # 数据丢失概率
    OUTLIER_PROBABILITY = 0.005       # 异常值概率

config = SimulationConfig()

# --- 2. 高级负荷模型 ---
class LoadProfileGenerator:
    """生成更真实的负荷曲线"""
    
    def __init__(self, base_loads):
        self.base_loads = base_loads
        self.time_of_day = 0
        self.day_of_year = 0
        
    def generate_load_profile(self, time_step):
        """生成基于时间的负荷分布"""
        # 日负荷曲线 (24小时周期)
        hour_of_day = (time_step * 0.1) % 24  # 假设每个时间步代表6分钟
        daily_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # 周负荷曲线
        day_of_week = (time_step // 240) % 7  # 假设一天240个时间步
        weekly_factor = 1.0 if day_of_week < 5 else 0.85  # 工作日vs周末
        
        # 季节性变化
        day_of_year = (time_step // 1680) % 365  # 一周1680个时间步
        seasonal_factor = 0.9 + 0.2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # 随机波动
        stochastic_factor = np.random.normal(1.0, 0.05, len(self.base_loads))
        
        # 组合所有因子
        total_factor = daily_factor * weekly_factor * seasonal_factor * stochastic_factor
        
        return self.base_loads * total_factor

# --- 3. 高级发电机模型 ---
class GeneratorController:
    """模拟更真实的发电机行为"""
    
    def __init__(self, generators):
        self.generators = generators
        self.ramp_rates = np.random.uniform(5, 20, len(generators))  # MW/min
        self.target_outputs = generators['p_mw'].values.copy()
        self.current_outputs = generators['p_mw'].values.copy()
        
    def update_generation(self, time_step, load_demand):
        """更新发电机出力"""
        # 计算总负荷需求
        total_demand = np.sum(load_demand)
        
        # AGC (自动发电控制) 模拟
        generation_error = total_demand - np.sum(self.current_outputs)
        
        # 按容量比例分配功率调整
        capacities = self.generators['max_p_mw'].values
        capacity_ratios = capacities / np.sum(capacities)
        power_adjustments = generation_error * capacity_ratios
        
        # 更新目标出力
        self.target_outputs = np.clip(
            self.current_outputs + power_adjustments,
            self.generators['min_p_mw'].values,
            self.generators['max_p_mw'].values
        )
        
        # 考虑爬坡限制
        max_ramp = self.ramp_rates * 0.1  # 假设每个时间步是6分钟
        ramp_limited_outputs = np.clip(
            self.target_outputs,
            self.current_outputs - max_ramp,
            self.current_outputs + max_ramp
        )
        
        # 添加可再生能源波动
        if config.RENEWABLE_GENERATION:
            renewable_indices = np.random.choice(len(self.generators), 
                                               size=max(1, len(self.generators)//3), 
                                               replace=False)
            renewable_factors = np.random.uniform(0.5, 1.5, len(renewable_indices))
            ramp_limited_outputs[renewable_indices] *= renewable_factors
        
        self.current_outputs = ramp_limited_outputs
        return self.current_outputs

# --- 4. 增强的故障注入系统 ---
class AdvancedFaultInjector:
    """高级故障注入器，支持复杂故障模式"""
    
    def __init__(self, net):
        self.net = net
        self.active_faults = {}
        self.degradation_states = {}
        self.cascading_events = []
        
        # 初始化设备健康状态
        self.init_equipment_health()
        
    def init_equipment_health(self):
        """初始化设备健康状态"""
        for line_idx in self.net.line.index:
            self.degradation_states[f'line_{line_idx}'] = {
                'health': np.random.uniform(0.8, 1.0),
                'impedance_drift': 0.0,
                'insulation_level': np.random.uniform(0.9, 1.0)
            }
        
        for gen_idx in self.net.gen.index:
            self.degradation_states[f'gen_{gen_idx}'] = {
                'health': np.random.uniform(0.85, 1.0),
                'efficiency': np.random.uniform(0.9, 0.95)
            }
    
    def update_degradation(self, time_step):
        """更新设备劣化状态"""
        for equipment, state in self.degradation_states.items():
            # 随机劣化
            degradation = np.random.exponential(config.DEGRADATION_RATE)
            state['health'] = max(0.1, state['health'] - degradation)
            
            # 特定设备的劣化
            if 'line_' in equipment:
                state['impedance_drift'] += np.random.normal(0, 0.0001)
                state['insulation_level'] = max(0.1, 
                    state['insulation_level'] - np.random.exponential(0.0002))
    
    def apply_complex_fault(self, fault_type, fault_info=None):
        """应用复杂故障"""
        fault_id = len(self.active_faults)
        
        if fault_type == "cascading_failure":
            return self._apply_cascading_fault(fault_id)
        elif fault_type == "intermittent_fault":
            return self._apply_intermittent_fault(fault_id)
        elif fault_type == "line_impedance_drift":
            return self._apply_impedance_drift(fault_id)
        elif fault_type == "insulation_degradation":
            return self._apply_insulation_degradation(fault_id)
        elif fault_type == "protection_malfunction":
            return self._apply_protection_malfunction(fault_id)
        elif fault_type == "voltage_regulator_fault":
            return self._apply_voltage_regulator_fault(fault_id)
        elif fault_type == "generator_instability":
            return self._apply_generator_instability(fault_id)
        elif fault_type == "transformer_outage":
            return self._apply_transformer_outage(fault_id)
        elif fault_type == "bus_fault":
            return self._apply_bus_fault(fault_id)
        else:
            # 调用原有的简单故障
            return self._apply_simple_fault(fault_type, fault_id)
    
    def _apply_cascading_fault(self, fault_id):
        """应用连锁故障"""
        # 选择初始故障线路
        available_lines = self.net.line[self.net.line.in_service == True].index
        if len(available_lines) == 0:
            return None
            
        initial_line = np.random.choice(available_lines)
        affected_lines = [initial_line]
        
        # 模拟连锁效应
        for _ in range(np.random.randint(1, 4)):  # 最多影响4条线路
            if np.random.random() < config.CASCADING_PROBABILITY:
                remaining_lines = [l for l in available_lines if l not in affected_lines]
                if remaining_lines:
                    next_line = np.random.choice(remaining_lines)
                    affected_lines.append(next_line)
        
        # 断开所有受影响的线路
        for line_idx in affected_lines:
            self.net.line.loc[line_idx, "in_service"] = False
        
        return {
            "type": "cascading_failure",
            "fault_id": fault_id,
            "affected_elements": affected_lines,
            "severity": len(affected_lines) / len(available_lines)
        }
    
    def _apply_intermittent_fault(self, fault_id):
        """应用间歇性故障"""
        available_lines = self.net.line[self.net.line.in_service == True].index
        if len(available_lines) == 0:
            return None
            
        line_to_affect = np.random.choice(available_lines)
        
        # 间歇性故障通过修改线路参数实现
        original_r = self.net.line.loc[line_to_affect, "r_ohm_per_km"]
        original_x = self.net.line.loc[line_to_affect, "x_ohm_per_km"]
        
        # 增加阻抗来模拟接触不良
        fault_factor = np.random.uniform(1.5, 3.0)
        self.net.line.loc[line_to_affect, "r_ohm_per_km"] *= fault_factor
        self.net.line.loc[line_to_affect, "x_ohm_per_km"] *= fault_factor
        
        return {
            "type": "intermittent_fault",
            "fault_id": fault_id,
            "affected_elements": [line_to_affect],
            "original_r": original_r,
            "original_x": original_x,
            "fault_factor": fault_factor
        }
    
    def _apply_impedance_drift(self, fault_id):
        """应用阻抗漂移故障（渐变性）"""
        available_lines = self.net.line[self.net.line.in_service == True].index
        if len(available_lines) == 0:
            return None
            
        line_to_affect = np.random.choice(available_lines)
        
        # 获取当前劣化状态
        equipment_key = f'line_{line_to_affect}'
        if equipment_key in self.degradation_states:
            drift = self.degradation_states[equipment_key]['impedance_drift']
            
            # 应用累积的阻抗漂移
            drift_factor = 1.0 + abs(drift) * 10  # 放大漂移效应
            self.net.line.loc[line_to_affect, "r_ohm_per_km"] *= drift_factor
            self.net.line.loc[line_to_affect, "x_ohm_per_km"] *= drift_factor
        
        return {
            "type": "line_impedance_drift",
            "fault_id": fault_id,
            "affected_elements": [line_to_affect],
            "drift_amount": drift if equipment_key in self.degradation_states else 0
        }
    
    def _apply_generator_instability(self, fault_id):
        """应用发电机不稳定故障"""
        available_gens = self.net.gen[self.net.gen.in_service == True].index
        if len(available_gens) == 0:
            return None
            
        gen_to_affect = np.random.choice(available_gens)
        
        # 模拟发电机输出不稳定
        original_p = self.net.gen.loc[gen_to_affect, "p_mw"]
        instability_factor = np.random.uniform(0.3, 1.7)  # 大幅波动
        self.net.gen.loc[gen_to_affect, "p_mw"] *= instability_factor
        
        return {
            "type": "generator_instability",
            "fault_id": fault_id,
            "affected_elements": [gen_to_affect],
            "original_p": original_p,
            "instability_factor": instability_factor
        }
    
    def _apply_transformer_outage(self, fault_id):
        """应用变压器故障"""
        if hasattr(self.net, 'trafo') and len(self.net.trafo) > 0:
            available_trafos = self.net.trafo[self.net.trafo.in_service == True].index
            if len(available_trafos) > 0:
                trafo_to_trip = np.random.choice(available_trafos)
                self.net.trafo.loc[trafo_to_trip, "in_service"] = False
                
                return {
                    "type": "transformer_outage",
                    "fault_id": fault_id,
                    "affected_elements": [trafo_to_trip]
                }
        
        # 如果没有变压器，回退到线路故障
        return self._apply_simple_fault("single_line_outage", fault_id)
    
    def _apply_bus_fault(self, fault_id):
        """应用母线故障"""
        # 选择一个有连接的母线
        connected_buses = []
        for bus_idx in self.net.bus.index:
            if (any(self.net.line.from_bus == bus_idx) or 
                any(self.net.line.to_bus == bus_idx)):
                connected_buses.append(bus_idx)
        
        if not connected_buses:
            return None
        
        bus_to_fault = np.random.choice(connected_buses)
        
        # 断开与该母线相连的所有线路
        affected_lines = []
        for line_idx in self.net.line.index:
            if (self.net.line.loc[line_idx, "from_bus"] == bus_to_fault or
                self.net.line.loc[line_idx, "to_bus"] == bus_to_fault):
                self.net.line.loc[line_idx, "in_service"] = False
                affected_lines.append(line_idx)
        
        # 断开该母线上的发电机
        affected_gens = []
        for gen_idx in self.net.gen.index:
            if self.net.gen.loc[gen_idx, "bus"] == bus_to_fault:
                self.net.gen.loc[gen_idx, "in_service"] = False
                affected_gens.append(gen_idx)
        
        return {
            "type": "bus_fault",
            "fault_id": fault_id,
            "affected_bus": bus_to_fault,
            "affected_lines": affected_lines,
            "affected_generators": affected_gens
        }
    
    def _apply_protection_malfunction(self, fault_id):
        """应用保护误动作"""
        # 随机选择一条正常运行的线路进行误跳闸
        available_lines = self.net.line[self.net.line.in_service == True].index
        if len(available_lines) == 0:
            return None
            
        line_to_trip = np.random.choice(available_lines)
        self.net.line.loc[line_to_trip, "in_service"] = False
        
        return {
            "type": "protection_malfunction",
            "fault_id": fault_id,
            "affected_elements": [line_to_trip],
            "malfunction_type": "false_trip"
        }
    
    def _apply_voltage_regulator_fault(self, fault_id):
        """应用电压调节器故障"""
        # 模拟通过修改发电机的电压设定点
        available_gens = self.net.gen[self.net.gen.in_service == True].index
        if len(available_gens) == 0:
            return None
            
        gen_to_affect = np.random.choice(available_gens)
        original_vm_pu = self.net.gen.loc[gen_to_affect, "vm_pu"]
        
        # 电压调节器失效，导致电压偏离设定值
        voltage_deviation = np.random.uniform(-0.1, 0.1)  # ±10%
        self.net.gen.loc[gen_to_affect, "vm_pu"] = max(0.9, min(1.1, 
            original_vm_pu + voltage_deviation))
        
        return {
            "type": "voltage_regulator_fault",
            "fault_id": fault_id,
            "affected_elements": [gen_to_affect],
            "original_vm_pu": original_vm_pu,
            "voltage_deviation": voltage_deviation
        }
    
    def _apply_simple_fault(self, fault_type, fault_id):
        """应用简单故障（原有逻辑）"""
        if fault_type == "single_line_outage":
            available_lines = self.net.line[self.net.line.in_service == True].index
            if len(available_lines) > 0:
                line_to_trip = np.random.choice(available_lines)
                self.net.line.loc[line_to_trip, "in_service"] = False
                return {
                    "type": fault_type,
                    "fault_id": fault_id,
                    "affected_elements": [line_to_trip]
                }
        
        elif fault_type == "double_line_outage":
            available_lines = self.net.line[self.net.line.in_service == True].index
            if len(available_lines) >= 2:
                lines_to_trip = np.random.choice(available_lines, size=2, replace=False)
                self.net.line.loc[lines_to_trip, "in_service"] = False
                return {
                    "type": fault_type,
                    "fault_id": fault_id,
                    "affected_elements": lines_to_trip.tolist()
                }
        
        elif fault_type == "generator_outage":
            available_gens = self.net.gen[self.net.gen.in_service == True].index
            if len(available_gens) > 0:
                gen_to_trip = np.random.choice(available_gens)
                self.net.gen.loc[gen_to_trip, "in_service"] = False
                return {
                    "type": fault_type,
                    "fault_id": fault_id,
                    "affected_elements": [gen_to_trip]
                }
        
        # 负荷故障逻辑保持不变
        elif fault_type in ["load_spike_moderate", "load_spike_severe"]:
            if len(self.net.load) > 0:
                spike_factor_range = (1.5, 2.5) if fault_type == "load_spike_moderate" else (3.0, 5.0)
                num_loads = max(1, int(len(self.net.load) * (0.2 if fault_type == "load_spike_moderate" else 0.1)))
                
                loads_to_spike = np.random.choice(self.net.load.index, size=num_loads, replace=False)
                spike_factor = np.random.uniform(*spike_factor_range, size=num_loads)
                
                original_p = self.net.load.loc[loads_to_spike, "p_mw"].values.copy()
                original_q = self.net.load.loc[loads_to_spike, "q_mvar"].values.copy()
                
                self.net.load.loc[loads_to_spike, "p_mw"] *= spike_factor
                self.net.load.loc[loads_to_spike, "q_mvar"] *= spike_factor
                
                return {
                    "type": fault_type,
                    "fault_id": fault_id,
                    "affected_elements": loads_to_spike.tolist(),
                    "original_p": original_p,
                    "original_q": original_q,
                    "spike_factor": spike_factor
                }
        
        return None
    
    def _apply_insulation_degradation(self, fault_id):
        """应用绝缘性能退化故障（渐变性）"""
        available_lines = self.net.line[self.net.line.in_service == True].index
        if len(available_lines) == 0:
            return None
            
        line_to_affect = np.random.choice(available_lines)
        
        # 获取当前劣化状态
        equipment_key = f'line_{line_to_affect}'
        if equipment_key in self.degradation_states:
            insulation_level = self.degradation_states[equipment_key]['insulation_level']
            
            # 应用累积的绝缘性能下降
            degradation_factor = 1.0 - insulation_level
            self.net.line.loc[line_to_affect, "x_ohm_per_km"] *= (1.0 + degradation_factor * 2)
            self.net.line.loc[line_to_affect, "r_ohm_per_km"] *= (1.0 + degradation_factor)
        
        return {
            "type": "insulation_degradation",
            "fault_id": fault_id,
            "affected_elements": [line_to_affect],
            "insulation_level": insulation_level if equipment_key in self.degradation_states else 1.0
        }

# --- 5. 测量噪声和数据质量模型 ---
class MeasurementSystem:
    """模拟真实的测量系统，包括噪声和数据丢失"""
    
    @staticmethod
    def add_measurement_noise(value, noise_std=None):
        """添加测量噪声"""
        if noise_std is None:
            noise_std = config.MEASUREMENT_NOISE_STD
        
        if np.isnan(value):
            return value
        
        noise = np.random.normal(0, noise_std * abs(value))
        return value + noise
    
    @staticmethod
    def simulate_data_loss(value):
        """模拟数据丢失"""
        if np.random.random() < config.MISSING_DATA_PROBABILITY:
            return np.nan
        return value
    
    @staticmethod
    def simulate_outliers(value):
        """模拟测量异常值"""
        if np.random.random() < config.OUTLIER_PROBABILITY:
            outlier_factor = np.random.choice([-1, 1]) * np.random.uniform(2, 5)
            return value * outlier_factor
        return value
    
    @staticmethod
    def process_measurement(value, measurement_type='general'):
        """处理测量值，应用噪声、丢失和异常值"""
        if np.isnan(value):
            return value
        
        # 选择合适的噪声标准差
        if measurement_type == 'voltage':
            noise_std = config.VOLTAGE_NOISE_STD
        elif measurement_type == 'power':
            noise_std = config.POWER_NOISE_STD
        else:
            noise_std = config.MEASUREMENT_NOISE_STD
        
        # 应用处理步骤
        value = MeasurementSystem.add_measurement_noise(value, noise_std)
        value = MeasurementSystem.simulate_outliers(value)
        value = MeasurementSystem.simulate_data_loss(value)
        
        return value

# --- 6. 主仿真函数（增强版） ---
def generate_complex_dynamic_dataset():
    """生成复杂动态数据集的主函数"""
    
    for net_name, net_func in NETWORKS_TO_SIMULATE.items():
        print(f"\n{'='*30} Generating complex data for: {net_name} {'='*30}")
        net_output_dir = os.path.join(OUTPUT_DIR, net_name)
        os.makedirs(net_output_dir, exist_ok=True)

        for sample_idx in tqdm(range(SAMPLES_PER_CASE), desc=f"Complex simulations for {net_name}"):
            base_net = net_func()
            
            # 记录原始网络状态
            original_load_p = base_net.load["p_mw"].values.copy()
            original_load_q = base_net.load["q_mvar"].values.copy()
            
            # 创建仿真所需的各种控制器
            net = copy.deepcopy(base_net)
            load_generator = LoadProfileGenerator(original_load_p)
            gen_controller = GeneratorController(net.gen)
            fault_injector = AdvancedFaultInjector(net)
            
            all_time_step_records = []
            
            # 生成更复杂的故障序列
            num_faults = np.random.poisson(2)  # 平均2个故障，但可能更多或更少
            fault_schedule = []
            
            for fault_id in range(num_faults):
                # 更智能的故障类型选择
                fault_weights = {
                    "single_line_outage": 0.15,
                    "double_line_outage": 0.08,
                    "generator_outage": 0.12,
                    "transformer_outage": 0.10,
                    "bus_fault": 0.05,
                    "load_spike_moderate": 0.12,
                    "load_spike_severe": 0.08,
                    "generator_instability": 0.10,
                    "line_impedance_drift": 0.05,
                    "insulation_degradation": 0.03,
                    "protection_malfunction": 0.04,
                    "voltage_regulator_fault": 0.04,
                    "cascading_failure": 0.02,
                    "intermittent_fault": 0.02,
                }
                
                fault_types = list(fault_weights.keys())
                weights = list(fault_weights.values())
                fault_type = np.random.choice(fault_types, p=weights)
                
                start_time = np.random.randint(10, SIMULATION_TIME_STEPS - 30)
                
                # 不同故障类型的持续时间分布
                if fault_type in ["cascading_failure", "bus_fault"]:
                    duration = np.random.randint(5, 15)  # 严重故障持续时间较短
                elif fault_type in ["line_impedance_drift", "insulation_degradation"]:
                    duration = np.random.randint(50, 100)  # 渐变故障持续时间较长
                elif fault_type == "intermittent_fault":
                    duration = np.random.randint(2, 8)   # 间歇性故障持续时间很短
                else:
                    duration = np.random.randint(5, 25)  # 一般故障
                
                end_time = min(start_time + duration, SIMULATION_TIME_STEPS - 5)
                fault_schedule.append((fault_id, fault_type, start_time, end_time))
            
            fault_schedule.sort(key=lambda x: x[2])
            
            # 初始化运行模式
            current_mode = "normal"
            mode_change_times = sorted(np.random.choice(
                range(20, SIMULATION_TIME_STEPS-20), 
                size=np.random.randint(0, 3), 
                replace=False
            ))
            
            # 时序仿真循环
            for time_step in range(SIMULATION_TIME_STEPS):
                
                # --- 运行模式管理 ---
                if time_step in mode_change_times:
                    if np.random.random() < config.MAINTENANCE_MODE_PROBABILITY:
                        current_mode = "maintenance"
                    elif np.random.random() < config.EMERGENCY_MODE_PROBABILITY:
                        current_mode = "emergency"
                    else:
                        current_mode = "normal"
                
                # --- 设备劣化更新 ---
                fault_injector.update_degradation(time_step)
                
                # --- 故障管理 ---
                faults_to_start = [f for f in fault_schedule if f[2] == time_step]
                for fault_id, fault_type, start_time, end_time in faults_to_start:
                    fault_info = fault_injector.apply_complex_fault(fault_type)
                    if fault_info:
                        fault_injector.active_faults[fault_id] = fault_info
                
                faults_to_end = [f for f in fault_schedule if f[3] == time_step]
                for fault_id, fault_type, start_time, end_time in faults_to_end:
                    if fault_id in fault_injector.active_faults:
                        # 故障恢复逻辑（简化）
                        del fault_injector.active_faults[fault_id]
                
                # --- 高级负荷模型 ---
                if config.DAILY_LOAD_VARIATION:
                    load_profile = load_generator.generate_load_profile(time_step)
                    
                    # 检查哪些负荷正在经历故障
                    loads_with_faults = set()
                    for fault_info in fault_injector.active_faults.values():
                        if fault_info["type"] in ["load_spike_moderate", "load_spike_severe"]:
                            loads_with_faults.update(fault_info["affected_elements"])
                    
                    # 只对没有故障的负荷应用正常负荷曲线
                    for load_idx in net.load.index:
                        if load_idx not in loads_with_faults:
                            net.load.loc[load_idx, "p_mw"] = load_profile[load_idx]
                            net.load.loc[load_idx, "q_mvar"] = load_profile[load_idx] * 0.3  # 假设功率因数
                
                # --- 发电机控制 ---
                if config.GENERATOR_RAMPING:
                    current_load = net.load["p_mw"].sum()
                    new_outputs = gen_controller.update_generation(time_step, net.load["p_mw"].values)
                    
                    # 更新发电机出力（仅对在线发电机）
                    for idx, gen_idx in enumerate(net.gen.index):
                        if net.gen.loc[gen_idx, "in_service"]:
                            net.gen.loc[gen_idx, "p_mw"] = new_outputs[idx]
                
                # --- 潮流计算 ---
                try:
                    # 根据运行模式调整计算参数
                    if current_mode == "emergency":
                        pp.runpp(net, algorithm="nr", enforce_q_lims=False, numba=True, max_iteration=30)
                    else:
                        pp.runpp(net, algorithm="nr", enforce_q_lims=True, numba=True)
                    converged = True
                except (pp.LoadflowNotConverged, Exception):
                    converged = False

                # --- 特征提取和测量处理 ---
                features = {
                    "sample_id": f"{net_name}_{sample_idx}",
                    "time_step": time_step,
                    "network": net_name,
                    "converged": converged,
                    "operation_mode": current_mode,
                    "num_active_faults": len(fault_injector.active_faults),
                }

                # 记录故障信息
                for fault_type in FAULT_TYPES:
                    if fault_type == "normal":
                        features[fault_type] = len(fault_injector.active_faults) == 0
                    else:
                        features[fault_type] = any(
                            fault_info["type"] == fault_type 
                            for fault_info in fault_injector.active_faults.values()
                        )

                if converged:
                    # 母线特征（加入测量噪声）
                    for bus_idx in net.res_bus.index:
                        vm_pu = MeasurementSystem.process_measurement(
                            net.res_bus.vm_pu.at[bus_idx], 'voltage'
                        )
                        va_degree = MeasurementSystem.process_measurement(
                            net.res_bus.va_degree.at[bus_idx]
                        )
                        features[f"bus_{bus_idx}_vm_pu"] = vm_pu
                        features[f"bus_{bus_idx}_va_degree"] = va_degree

                    # 线路特征
                    for line_idx in net.res_line.index:
                        loading = MeasurementSystem.process_measurement(
                            net.res_line.loading_percent.at[line_idx]
                        )
                        features[f"line_{line_idx}_loading_percent"] = loading
                        features[f"line_{line_idx}_in_service"] = net.line.in_service.at[line_idx]

                    # 发电机特征
                    for gen_idx in net.res_gen.index:
                        p_mw = MeasurementSystem.process_measurement(
                            net.res_gen.p_mw.at[gen_idx], 'power'
                        )
                        q_mvar = MeasurementSystem.process_measurement(
                            net.res_gen.q_mvar.at[gen_idx], 'power'
                        )
                        features[f"gen_{gen_idx}_p_mw"] = p_mw
                        features[f"gen_{gen_idx}_q_mvar"] = q_mvar
                        features[f"gen_{gen_idx}_in_service"] = net.gen.in_service.at[gen_idx]

                    # 负荷特征
                    for load_idx in net.load.index:
                        p_mw = MeasurementSystem.process_measurement(
                            net.load.p_mw.at[load_idx], 'power'
                        )
                        q_mvar = MeasurementSystem.process_measurement(
                            net.load.q_mvar.at[load_idx], 'power'
                        )
                        features[f"load_{load_idx}_p_mw"] = p_mw
                        features[f"load_{load_idx}_q_mvar"] = q_mvar

                    # 总体特征
                    line_losses = net.res_line.pl_mw.sum()
                    trafo_losses = (net.res_trafo.pl_mw.sum() 
                                  if hasattr(net, "res_trafo") and len(net.res_trafo) > 0 
                                  else 0.0)
                    
                    total_loss = MeasurementSystem.process_measurement(
                        line_losses + trafo_losses, 'power'
                    )
                    total_gen = MeasurementSystem.process_measurement(
                        net.res_gen.p_mw.sum(), 'power'
                    )
                    total_load = MeasurementSystem.process_measurement(
                        net.load.p_mw.sum(), 'power'
                    )
                    
                    features["total_loss_mw"] = total_loss
                    features["total_generation_mw"] = total_gen
                    features["total_load_mw"] = total_load

                else:
                    # 不收敛时填充 NaN（模拟测量失败）
                    for bus_idx in net.bus.index:
                        features[f"bus_{bus_idx}_vm_pu"] = np.nan
                        features[f"bus_{bus_idx}_va_degree"] = np.nan
                    for line_idx in net.line.index:
                        features[f"line_{line_idx}_loading_percent"] = np.nan
                        features[f"line_{line_idx}_in_service"] = net.line.in_service.at[line_idx]
                    for gen_idx in net.gen.index:
                        features[f"gen_{gen_idx}_p_mw"] = np.nan
                        features[f"gen_{gen_idx}_q_mvar"] = np.nan
                        features[f"gen_{gen_idx}_in_service"] = net.gen.in_service.at[gen_idx]
                    for load_idx in net.load.index:
                        features[f"load_{load_idx}_p_mw"] = net.load.p_mw.at[load_idx]
                        features[f"load_{load_idx}_q_mvar"] = net.load.q_mvar.at[load_idx]
                    features["total_loss_mw"] = np.nan
                    features["total_generation_mw"] = np.nan
                    features["total_load_mw"] = net.load.p_mw.sum()

                all_time_step_records.append(features)

            # 保存到 CSV 文件
            df = pd.DataFrame(all_time_step_records)
            output_path = os.path.join(net_output_dir, f"{net_name}_complex_sample_{sample_idx}.csv")
            df.to_csv(output_path, index=False)

    print("\n--- All complex datasets generated successfully! ---")

# --- 7. 增强的数据集分析函数 ---
def analyze_complex_dataset(output_dir):
    """分析生成的复杂数据集"""
    print(f"\n{'='*30} Complex Dataset Analysis {'='*30}")
    
    for net_name in NETWORKS_TO_SIMULATE.keys():
        net_dir = os.path.join(output_dir, net_name)
        if not os.path.exists(net_dir):
            continue
            
        print(f"\n--- {net_name} ---")
        csv_files = [f for f in os.listdir(net_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found")
            continue
            
        # 读取所有文件进行综合分析
        all_data = []
        for csv_file in csv_files[:5]:  # 分析前5个文件以节省时间
            df = pd.read_csv(os.path.join(net_dir, csv_file))
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"Total time steps analyzed: {len(combined_df)}")
        print(f"Number of simulations: {len(csv_files)}")
        print(f"Features per time step: {len(combined_df.columns)}")
        
        # 分析故障分布
        fault_columns = [col for col in combined_df.columns if col in FAULT_TYPES]
        print(f"\nFault distribution:")
        
        total_steps = len(combined_df)
        for fault_type in fault_columns:
            if fault_type in combined_df.columns:
                fault_count = combined_df[fault_type].sum()
                print(f"  {fault_type:25}: {fault_count:6} steps ({fault_count/total_steps:6.1%})")
        
        # 分析收敛性
        convergence_rate = combined_df['converged'].mean()
        print(f"\nConvergence rate: {convergence_rate:.2%}")
        
        # 分析数据质量
        nan_columns = combined_df.select_dtypes(include=[np.number]).columns
        nan_rates = combined_df[nan_columns].isnull().mean()
        high_nan_cols = nan_rates[nan_rates > 0.01].sort_values(ascending=False)
        
        if len(high_nan_cols) > 0:
            print(f"\nColumns with >1% missing data:")
            for col, rate in high_nan_cols.head(10).items():
                print(f"  {col:30}: {rate:.1%}")
        
        # 分析运行模式分布
        if 'operation_mode' in combined_df.columns:
            mode_counts = combined_df['operation_mode'].value_counts()
            print(f"\nOperation mode distribution:")
            for mode, count in mode_counts.items():
                print(f"  {mode:15}: {count:6} steps ({count/total_steps:6.1%})")
        
        # 分析多故障并发
        if 'num_active_faults' in combined_df.columns:
            fault_dist = combined_df['num_active_faults'].value_counts().sort_index()
            print(f"\nConcurrent faults distribution:")
            for num_faults, count in fault_dist.items():
                print(f"  {num_faults} faults: {count:6} steps ({count/total_steps:6.1%})")

# --- 8. 运行主函数 ---
if __name__ == "__main__":
    # 设置随机种子以便复现
    np.random.seed(42)
    
    # 生成复杂数据集
    generate_complex_dynamic_dataset()
    
    # 分析数据集
    analyze_complex_dataset(OUTPUT_DIR)
    
    print(f"\n{'='*50}")
    print("Complex dataset generation completed!")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("Key improvements:")
    print("- 15 different fault types including cascading and intermittent faults")
    print("- Realistic load profiles with daily/seasonal variations")
    print("- Generator ramping and renewable energy fluctuations")
    print("- Measurement noise and data quality issues")
    print("- Equipment degradation and health monitoring")
    print("- Multiple operation modes (normal/maintenance/emergency)")
    print("- Advanced fault interactions and dependencies")
    print(f"{'='*50}")