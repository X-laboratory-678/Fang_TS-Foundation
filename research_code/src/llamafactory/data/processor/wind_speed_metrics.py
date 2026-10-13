# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wind Speed Prediction Evaluation Metrics
专门用于风速预测任务的评估指标模块
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


class WindSpeedMetrics:
    """
    风速预测专用评估指标类
    包含时间序列预测常用的各种评估指标
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        初始化评估指标计算器
        
        Args:
            epsilon: 防止除零的小常数
        """
        self.epsilon = epsilon
    
    def mae(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算平均绝对误差 (Mean Absolute Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            MAE值
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        return float(mean_absolute_error(y_true, y_pred))
    
    def mse(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算均方误差 (Mean Squared Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            MSE值
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        return float(mean_squared_error(y_true, y_pred))
    
    def rmse(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算均方根误差 (Root Mean Squared Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            RMSE值
        """
        mse_val = self.mse(y_true, y_pred)
        return float(np.sqrt(mse_val))
    
    def mape(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算平均绝对百分比误差 (Mean Absolute Percentage Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            MAPE值 (百分比)
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        
        # 避免除零
        mask = np.abs(y_true) > self.epsilon
        if not np.any(mask):
            warnings.warn("所有真实值都接近零，MAPE可能不准确")
            return float('inf')
        
        mape_val = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape_val)
    
    def smape(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算对称平均绝对百分比误差 (Symmetric Mean Absolute Percentage Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            SMAPE值 (百分比)
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator > self.epsilon
        
        if not np.any(mask):
            return 0.0
        
        smape_val = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        return float(smape_val)
    
    def r2(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算决定系数 (R-squared)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            R²值
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        return float(r2_score(y_true, y_pred))
    
    def direction_accuracy(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算方向准确率（预测变化方向的准确性）
        
        Args:
            y_true: 真实值序列
            y_pred: 预测值序列
            
        Returns:
            方向准确率 (0-1)
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        
        if len(y_true) < 2:
            return 0.0
        
        # 计算变化方向
        true_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        
        # 计算方向一致性
        direction_match = np.sign(true_diff) == np.sign(pred_diff)
        
        # 忽略变化为零的情况
        non_zero_mask = np.abs(true_diff) > self.epsilon
        if not np.any(non_zero_mask):
            return 1.0  # 如果真实值没有变化，认为方向预测完全正确
        
        accuracy = np.mean(direction_match[non_zero_mask])
        return float(accuracy)
    
    def peak_accuracy(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor], 
                     threshold: float = 0.1) -> float:
        """
        计算峰值预测准确率
        
        Args:
            y_true: 真实值序列
            y_pred: 预测值序列
            threshold: 峰值检测阈值（相对于标准差的倍数）
            
        Returns:
            峰值准确率 (0-1)
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        
        if len(y_true) < 3:
            return 0.0
        
        # 检测真实峰值
        true_std = np.std(y_true)
        true_mean = np.mean(y_true)
        true_peaks = self._detect_peaks(y_true, true_mean + threshold * true_std)
        
        # 检测预测峰值
        pred_std = np.std(y_pred)
        pred_mean = np.mean(y_pred)
        pred_peaks = self._detect_peaks(y_pred, pred_mean + threshold * pred_std)
        
        if len(true_peaks) == 0:
            return 1.0 if len(pred_peaks) == 0 else 0.0
        
        # 计算峰值匹配度（允许±1的位置误差）
        matches = 0
        for true_peak in true_peaks:
            for pred_peak in pred_peaks:
                if abs(true_peak - pred_peak) <= 1:
                    matches += 1
                    break
        
        accuracy = matches / len(true_peaks)
        return float(accuracy)
    
    def wape(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """
        计算加权平均绝对百分比误差 (Weighted Absolute Percentage Error)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            WAPE值 (百分比)
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        
        sum_abs_true = np.sum(np.abs(y_true))
        if sum_abs_true < self.epsilon:
            return float('inf')
        
        wape_val = np.sum(np.abs(y_true - y_pred)) / sum_abs_true * 100
        return float(wape_val)
    
    def quantile_loss(self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor], 
                     quantile: float = 0.5) -> float:
        """
        计算分位数损失
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            quantile: 分位数 (0-1)
            
        Returns:
            分位数损失值
        """
        y_true, y_pred = self._to_numpy(y_true, y_pred)
        
        errors = y_true - y_pred
        loss = np.maximum(quantile * errors, (quantile - 1) * errors)
        return float(np.mean(loss))
    
    def compute_all_metrics(self, y_true: Union[np.ndarray, torch.Tensor], 
                           y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        try:
            metrics['mae'] = self.mae(y_true, y_pred)
            metrics['mse'] = self.mse(y_true, y_pred)
            metrics['rmse'] = self.rmse(y_true, y_pred)
            metrics['mape'] = self.mape(y_true, y_pred)
            metrics['smape'] = self.smape(y_true, y_pred)
            metrics['r2'] = self.r2(y_true, y_pred)
            metrics['wape'] = self.wape(y_true, y_pred)
            metrics['direction_accuracy'] = self.direction_accuracy(y_true, y_pred)
            metrics['peak_accuracy'] = self.peak_accuracy(y_true, y_pred)
            
            # 分位数损失
            for q in [0.1, 0.5, 0.9]:
                metrics[f'quantile_loss_{q}'] = self.quantile_loss(y_true, y_pred, q)
                
        except Exception as e:
            warnings.warn(f"计算指标时出现错误: {e}")
        
        return metrics
    
    def _to_numpy(self, y_true: Union[np.ndarray, torch.Tensor], 
                  y_pred: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将输入转换为numpy数组
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            转换后的numpy数组元组
        """
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if len(y_true) != len(y_pred):
            raise ValueError(f"真实值和预测值长度不匹配: {len(y_true)} vs {len(y_pred)}")
        
        return y_true, y_pred
    
    def _detect_peaks(self, data: np.ndarray, threshold: float) -> List[int]:
        """
        检测峰值位置
        
        Args:
            data: 数据序列
            threshold: 峰值阈值
            
        Returns:
            峰值位置列表
        """
        peaks = []
        for i in range(1, len(data) - 1):
            if (data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold):
                peaks.append(i)
        return peaks


class WindSpeedMetricsCallback:
    """
    用于训练过程中的指标回调类
    """
    
    def __init__(self, metrics_calculator: Optional[WindSpeedMetrics] = None):
        """
        初始化回调
        
        Args:
            metrics_calculator: 指标计算器实例
        """
        self.metrics_calculator = metrics_calculator or WindSpeedMetrics()
        self.history = []
    
    def on_epoch_end(self, y_true: Union[np.ndarray, torch.Tensor], 
                     y_pred: Union[np.ndarray, torch.Tensor], epoch: int) -> Dict[str, float]:
        """
        在每个epoch结束时计算指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            epoch: 当前epoch
            
        Returns:
            当前epoch的指标字典
        """
        metrics = self.metrics_calculator.compute_all_metrics(y_true, y_pred)
        metrics['epoch'] = epoch
        
        self.history.append(metrics)
        return metrics
    
    def get_best_epoch(self, metric: str = 'rmse', mode: str = 'min') -> Tuple[int, Dict[str, float]]:
        """
        获取最佳epoch
        
        Args:
            metric: 用于判断的指标名称
            mode: 'min' 或 'max'
            
        Returns:
            最佳epoch和对应的指标字典
        """
        if not self.history:
            raise ValueError("没有历史记录")
        
        if metric not in self.history[0]:
            raise ValueError(f"指标 '{metric}' 不存在")
        
        if mode == 'min':
            best_idx = min(range(len(self.history)), key=lambda i: self.history[i][metric])
        else:
            best_idx = max(range(len(self.history)), key=lambda i: self.history[i][metric])
        
        best_metrics = self.history[best_idx]
        return best_metrics['epoch'], best_metrics


# 便捷函数
def evaluate_wind_speed_prediction(y_true: Union[np.ndarray, torch.Tensor], 
                                  y_pred: Union[np.ndarray, torch.Tensor],
                                  detailed: bool = True) -> Dict[str, float]:
    """
    便捷的风速预测评估函数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        detailed: 是否返回详细指标
        
    Returns:
        评估指标字典
    """
    calculator = WindSpeedMetrics()
    
    if detailed:
        return calculator.compute_all_metrics(y_true, y_pred)
    else:
        # 只返回核心指标
        return {
            'mae': calculator.mae(y_true, y_pred),
            'rmse': calculator.rmse(y_true, y_pred),
            'mape': calculator.mape(y_true, y_pred),
            'r2': calculator.r2(y_true, y_pred)
        }


def print_metrics_summary(metrics: Dict[str, float], title: str = "评估结果") -> None:
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        title: 标题
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    # 核心指标
    core_metrics = ['mae', 'rmse', 'mape', 'r2']
    print("\n核心指标:")
    print("-" * 30)
    for metric in core_metrics:
        if metric in metrics:
            if metric == 'r2':
                print(f"{metric.upper():>10}: {metrics[metric]:>8.4f}")
            elif metric in ['mape']:
                print(f"{metric.upper():>10}: {metrics[metric]:>7.2f}%")
            else:
                print(f"{metric.upper():>10}: {metrics[metric]:>8.4f}")
    
    # 其他指标
    other_metrics = {k: v for k, v in metrics.items() if k not in core_metrics + ['epoch']}
    if other_metrics:
        print("\n其他指标:")
        print("-" * 30)
        for metric, value in other_metrics.items():
            if 'accuracy' in metric:
                print(f"{metric:>20}: {value:>7.2f}%")
            elif 'loss' in metric:
                print(f"{metric:>20}: {value:>8.4f}")
            elif metric in ['smape', 'wape']:
                print(f"{metric.upper():>20}: {value:>7.2f}%")
            else:
                print(f"{metric:>20}: {value:>8.4f}")
    
    print(f"{'='*50}\n")