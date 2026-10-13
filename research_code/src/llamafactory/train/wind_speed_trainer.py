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

import os
import math
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

from ..model.wind_speed_model import WindSpeedModel, WindSpeedConfig


logger = logging.getLogger(__name__)


class WindSpeedTrainingArguments(TrainingArguments):
    """
    风速预测专用的训练参数
    """
    def __init__(
        self,
        # 基础训练参数
        output_dir: str = "./wind_speed_output",
        overwrite_output_dir: bool = True,
        do_train: bool = True,
        do_eval: bool = True,
        do_predict: bool = False,
        evaluation_strategy: str = "steps",
        eval_steps: int = 500,
        per_device_train_batch_size: int = 32,
        per_device_eval_batch_size: int = 64,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        num_train_epochs: float = 10.0,
        max_steps: int = -1,
        lr_scheduler_type: str = "cosine_with_restarts",
        warmup_ratio: float = 0.1,
        warmup_steps: int = 0,
        logging_dir: Optional[str] = None,
        logging_strategy: str = "steps",
        logging_steps: int = 100,
        save_strategy: str = "steps",
        save_steps: int = 1000,
        save_total_limit: Optional[int] = 3,
        seed: int = 42,
        fp16: bool = False,
        bf16: bool = True,
        dataloader_drop_last: bool = True,
        dataloader_num_workers: int = 4,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_rmse",
        greater_is_better: bool = False,
        # 风速预测专用参数
        early_stopping_patience: int = 5,
        early_stopping_threshold: float = 0.001,
        use_cyclical_lr: bool = True,
        cyclical_lr_base_lr: float = 1e-5,
        cyclical_lr_max_lr: float = 5e-4,
        cyclical_lr_step_size: int = 2000,
        use_gradient_clipping: bool = True,
        gradient_clip_value: float = 1.0,
        **kwargs
    ):
        super().__init__(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
            evaluation_strategy=evaluation_strategy,
            eval_steps=eval_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            max_grad_norm=max_grad_norm,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            logging_dir=logging_dir,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            seed=seed,
            fp16=fp16,
            bf16=bf16,
            dataloader_drop_last=dataloader_drop_last,
            dataloader_num_workers=dataloader_num_workers,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            **kwargs
        )
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.use_cyclical_lr = use_cyclical_lr
        self.cyclical_lr_base_lr = cyclical_lr_base_lr
        self.cyclical_lr_max_lr = cyclical_lr_max_lr
        self.cyclical_lr_step_size = cyclical_lr_step_size
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clip_value = gradient_clip_value


class WindSpeedTrainer(Trainer):
    """
    风速预测专用的训练器
    """
    
    def __init__(
        self,
        model: WindSpeedModel,
        args: WindSpeedTrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        compute_metrics=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics or self.compute_wind_speed_metrics,
            **kwargs
        )
        
        self.best_metric = float('inf')
        self.patience_counter = 0
        
    def compute_wind_speed_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        计算风速预测专用的评估指标
        """
        predictions, labels = eval_pred
        
        # 确保预测和标签的形状一致
        if predictions.shape != labels.shape:
            logger.warning(f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}")
            min_len = min(len(predictions), len(labels))
            predictions = predictions[:min_len]
            labels = labels[:min_len]
        
        # 计算各种指标
        mae = mean_absolute_error(labels.flatten(), predictions.flatten())
        mse = mean_squared_error(labels.flatten(), predictions.flatten())
        rmse = math.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((labels - predictions) / (labels + 1e-8))) * 100
        
        # R² Score
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # 方向准确率 (预测趋势是否正确)
        if labels.shape[1] > 1:  # 多步预测
            pred_trend = np.diff(predictions, axis=1) > 0
            true_trend = np.diff(labels, axis=1) > 0
            direction_accuracy = np.mean(pred_trend == true_trend)
        else:
            direction_accuracy = 0.0
        
        # 峰值检测准确率
        pred_peaks = self._detect_peaks(predictions.flatten())
        true_peaks = self._detect_peaks(labels.flatten())
        peak_accuracy = len(set(pred_peaks) & set(true_peaks)) / (len(set(pred_peaks) | set(true_peaks)) + 1e-8)
        
        return {
            "eval_mae": mae,
            "eval_mse": mse,
            "eval_rmse": rmse,
            "eval_mape": mape,
            "eval_r2": r2,
            "eval_direction_accuracy": direction_accuracy,
            "eval_peak_accuracy": peak_accuracy,
        }
    
    def _detect_peaks(self, data: np.ndarray, threshold: float = 0.1) -> List[int]:
        """
        检测峰值点
        """
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1] and data[i] > threshold:
                peaks.append(i)
        return peaks
    
    def training_step(self, model, inputs):
        """
        自定义训练步骤
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失函数
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        # 自定义损失计算
        predictions = outputs.predictions
        
        print(f"Trainer using loss type: {model.config.loss_type}")  # 调试信息
        if model.config.loss_type == "mse":
            loss = torch.nn.functional.mse_loss(predictions, labels)
            print("Trainer using MSE loss")  # 调试信息
        elif model.config.loss_type == "mae":
            loss = torch.nn.functional.l1_loss(predictions, labels)
            print("Trainer using MAE loss")  # 调试信息
        elif model.config.loss_type == "huber":
            loss = torch.nn.functional.huber_loss(predictions, labels, delta=model.config.huber_delta)
            print("Trainer using Huber loss")  # 调试信息
        else:
            # 组合损失：MSE + MAE
            mse_loss = torch.nn.functional.mse_loss(predictions, labels)
            mae_loss = torch.nn.functional.l1_loss(predictions, labels)
            loss = 0.7 * mse_loss + 0.3 * mae_loss
            print("Trainer using combined loss")  # 调试信息
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        自定义评估方法，包含早停机制
        """
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 早停检查
        current_metric = eval_results.get(f"{metric_key_prefix}_rmse", float('inf'))
        
        if current_metric < self.best_metric - self.args.early_stopping_threshold:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(f"New best metric: {current_metric:.6f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement. Patience: {self.patience_counter}/{self.args.early_stopping_patience}")
            
            if self.patience_counter >= self.args.early_stopping_patience:
                logger.info("Early stopping triggered!")
                self.control.should_training_stop = True
        
        return eval_results
    
    def create_optimizer(self):
        """
        创建优化器，支持不同的学习率策略
        """
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() 
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() 
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        创建学习率调度器
        """
        if self.args.use_cyclical_lr:
            from torch.optim.lr_scheduler import CyclicLR
            
            self.lr_scheduler = CyclicLR(
                optimizer or self.optimizer,
                base_lr=self.args.cyclical_lr_base_lr,
                max_lr=self.args.cyclical_lr_max_lr,
                step_size_up=self.args.cyclical_lr_step_size,
                mode='triangular2',
                cycle_momentum=False
            )
        else:
            self.lr_scheduler = super().create_scheduler(num_training_steps, optimizer)
        
        return self.lr_scheduler
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        自定义日志记录
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        # 添加学习率信息
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            logs["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        
        # 添加梯度范数信息
        if hasattr(self, '_last_grad_norm'):
            logs["grad_norm"] = self._last_grad_norm
        
        super().log(logs)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        自定义检查点保存
        """
        checkpoint_folder = super()._save_checkpoint(model, trial, metrics)
        
        # 保存训练状态
        if checkpoint_folder is not None:
            training_state = {
                'best_metric': self.best_metric,
                'patience_counter': self.patience_counter,
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
            }
            
            torch.save(
                training_state, 
                os.path.join(checkpoint_folder, "training_state.pt")
            )
        
        return checkpoint_folder
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        从检查点加载训练状态
        """
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        
        # 加载训练状态
        training_state_path = os.path.join(resume_from_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu")
            self.best_metric = training_state.get('best_metric', float('inf'))
            self.patience_counter = training_state.get('patience_counter', 0)
            logger.info(f"Resumed training state: best_metric={self.best_metric}, patience={self.patience_counter}")


def create_wind_speed_trainer(
    model: WindSpeedModel,
    train_dataset,
    eval_dataset,
    output_dir: str = "./wind_speed_output",
    **kwargs
) -> WindSpeedTrainer:
    """
    创建风速预测训练器的便捷函数
    """
    training_args = WindSpeedTrainingArguments(
        output_dir=output_dir,
        **kwargs
    )
    
    trainer = WindSpeedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    return trainer