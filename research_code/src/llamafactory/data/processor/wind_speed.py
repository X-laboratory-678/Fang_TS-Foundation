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

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ...extras import logging
from .processor_utils import DatasetProcessor


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ...hparams import DataArguments
    from ..template import Template


logger = logging.get_logger(__name__)


@dataclass
class WindSpeedDatasetProcessor(DatasetProcessor):
    """
    专门用于风速预测的数据处理器
    支持时间序列数据的特征工程和预处理
    """
    
    def __init__(self, template, tokenizer, processor, data_args):
        super().__init__(template, tokenizer, processor, data_args)
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.sequence_length = getattr(data_args, 'sequence_length', 24)  # 默认24小时序列
        self.prediction_horizon = getattr(data_args, 'prediction_horizon', 1)  # 预测未来1小时
        self.feature_columns = getattr(data_args, 'feature_columns', [
            'wind_speed', 'temperature', 'humidity', 'pressure', 'wind_direction'
        ])
        
    def _create_sequences(self, data: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据的输入序列和目标序列
        
        Args:
            data: 特征数据 [时间步, 特征数]
            targets: 目标数据 [时间步]
            
        Returns:
            X: 输入序列 [样本数, 序列长度, 特征数]
            y: 目标序列 [样本数, 预测长度]
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # 输入序列
            seq_x = data[i:i + self.sequence_length]
            # 目标序列（预测未来的风速）
            seq_y = targets[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            
            X.append(seq_x)
            y.append(seq_y)
            
        return np.array(X), np.array(y)
    
    def _extract_features(self, examples: Dict[str, List[Any]]) -> tuple[np.ndarray, np.ndarray]:
        """
        从原始数据中提取特征
        
        Args:
            examples: 包含时间序列数据的字典
            
        Returns:
            features: 特征矩阵
            targets: 目标向量
        """
        features = []
        targets = []
        
        for i in range(len(examples['timestamp'])):
            # 提取基础特征
            feature_row = []
            for col in self.feature_columns:
                if col in examples:
                    feature_row.append(examples[col][i])
                else:
                    feature_row.append(0.0)  # 缺失值填充
            
            # 添加时间特征
            timestamp = examples['timestamp'][i]
            if isinstance(timestamp, str):
                import datetime
                dt = datetime.datetime.fromisoformat(timestamp)
                # 小时特征（周期性编码）
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                # 月份特征（周期性编码）
                month_sin = np.sin(2 * np.pi * dt.month / 12)
                month_cos = np.cos(2 * np.pi * dt.month / 12)
                # 星期特征（周期性编码）
                weekday_sin = np.sin(2 * np.pi * dt.weekday() / 7)
                weekday_cos = np.cos(2 * np.pi * dt.weekday() / 7)
                
                feature_row.extend([hour_sin, hour_cos, month_sin, month_cos, weekday_sin, weekday_cos])
            
            features.append(feature_row)
            targets.append(examples['wind_speed'][i])
        
        return np.array(features), np.array(targets)
    
    def _normalize_features(self, features: np.ndarray, targets: np.ndarray, fit_scaler: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        标准化特征和目标值
        
        Args:
            features: 特征矩阵
            targets: 目标向量
            fit_scaler: 是否拟合缩放器
            
        Returns:
            normalized_features: 标准化后的特征
            normalized_targets: 标准化后的目标
        """
        if fit_scaler:
            normalized_features = self.scaler.fit_transform(features)
            normalized_targets = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        else:
            normalized_features = self.scaler.transform(features)
            normalized_targets = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()
            
        return normalized_features, normalized_targets
    
    def _tokenize_numerical_data(self, sequences: np.ndarray, targets: np.ndarray) -> Dict[str, List[Any]]:
        """
        将数值数据转换为token格式
        
        Args:
            sequences: 输入序列 [样本数, 序列长度, 特征数]
            targets: 目标序列 [样本数, 预测长度]
            
        Returns:
            tokenized_data: 包含input_ids和labels的字典
        """
        model_inputs = defaultdict(list)
        
        for seq, target in zip(sequences, targets):
            # 将数值序列转换为文本表示
            input_text = "时间序列数据: "
            for t, features in enumerate(seq):
                input_text += f"时刻{t}: "
                for f, feature_val in enumerate(features):
                    input_text += f"特征{f}={feature_val:.4f} "
                input_text += "; "
            
            # 添加预测提示
            input_text += "预测未来风速: "
            
            # 目标文本
            target_text = f"{target[0]:.4f}"
            
            # 组合输入和目标
            full_text = input_text + target_text
            
            # 分词
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
            
            # 创建标签（只对目标部分计算损失）
            labels = [-100] * len(input_ids) + target_ids + [self.tokenizer.eos_token_id]
            input_ids = full_ids + [self.tokenizer.eos_token_id]
            
            # 截断到最大长度
            max_length = getattr(self.data_args, 'cutoff_len', 512)
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1] * len(input_ids))
        
        return model_inputs

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        预处理风速预测数据集
        
        Args:
            examples: 原始数据样本
            
        Returns:
            model_inputs: 处理后的模型输入
        """
        logger.info_rank0(f"Processing {len(examples['timestamp'])} wind speed samples...")
        
        # 提取特征
        features, targets = self._extract_features(examples)
        
        # 标准化
        normalized_features, normalized_targets = self._normalize_features(features, targets)
        
        # 创建时间序列
        sequences, sequence_targets = self._create_sequences(normalized_features, normalized_targets)
        
        logger.info_rank0(f"Created {len(sequences)} sequences with length {self.sequence_length}")
        
        # 转换为token格式
        model_inputs = self._tokenize_numerical_data(sequences, sequence_targets)
        
        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        """
        打印数据示例
        
        Args:
            example: 数据示例
        """
        if "input_ids" in example:
            input_text = self.tokenizer.decode(example["input_ids"], skip_special_tokens=True)
            print("=" * 50)
            print("Wind Speed Prediction Example:")
            print("=" * 50)
            print("Input:", input_text[:200] + "..." if len(input_text) > 200 else input_text)
            
            if "labels" in example:
                # 找到非-100的标签
                valid_labels = [token for token in example["labels"] if token != -100]
                if valid_labels:
                    label_text = self.tokenizer.decode(valid_labels, skip_special_tokens=True)
                    print("Target:", label_text)
            print("=" * 50)