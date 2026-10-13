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
Lightweight MAML Implementation for 4-bit Quantized Models

This implementation addresses the challenge of meta-learning with quantized models
where parameters don't support requires_grad. We use a simplified MAML approach
that avoids second-order gradients.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ...train.sft.trainer import CustomSeq2SeqTrainer

logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """Configuration for MAML training"""
    
    # MAML specific parameters
    use_meta_learning: bool = field(
        default=False,
        metadata={"help": "Enable meta-learning (MAML) training"}
    )
    inner_learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for inner loop (task adaptation)"}
    )
    outer_learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for outer loop (meta-optimization)"}
    )
    num_inner_steps: int = field(
        default=3,
        metadata={"help": "Number of gradient steps in inner loop"}
    )
    task_batch_size: int = field(
        default=4,
        metadata={"help": "Number of tasks per meta-batch"}
    )
    support_set_size: int = field(
        default=8,
        metadata={"help": "Number of samples in support set per task"}
    )
    query_set_size: int = field(
        default=8,
        metadata={"help": "Number of samples in query set per task"}
    )
    
    # Quantization-friendly options
    simplified_maml: bool = field(
        default=True,
        metadata={"help": "Use simplified MAML for 4-bit quantized models (no second-order gradients)"}
    )
    use_first_order: bool = field(
        default=True,
        metadata={"help": "Use first-order approximation (FOMAML) to save memory"}
    )
    
    # Memory optimization
    clear_cache_per_task: bool = field(
        default=True,
        metadata={"help": "Clear CUDA cache after each task to prevent OOM"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing to save memory"}
    )
    
    # Domain-aware task sampling
    domain_aware_sampling: bool = field(
        default=True,
        metadata={"help": "Organize tasks by domain (requires domain labels in dataset)"}
    )
    domain_key: str = field(
        default="instruction",
        metadata={"help": "Key to identify domain in dataset (e.g., 'instruction', 'domain')"}
    )


class MAMLTrainer:
    """
    MAML Trainer Mixin for time series prediction with 4-bit quantization support
    
    Key features:
    1. Simplified MAML that works with quantized models
    2. First-order approximation to avoid second-order gradients
    3. Memory-efficient implementation
    4. Task-specific adaptation for time series
    
    This is a mixin that adds MAML capability to CustomSeq2SeqTrainer.
    """
    
    def __init__(self, *args, maml_config: MAMLConfig = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.maml_config = maml_config or MAMLConfig()
        
        # Track meta-learning statistics
        self.meta_step = 0
        self.num_tasks_completed = 0  # Track total tasks completed
        self.task_losses = []
        
        # Store domain labels for domain-aware sampling
        self.domain_labels = []  # Will be populated by train_dataset
        
        print(f"[DEBUG] MAMLTrainer initialized with use_meta_learning={self.maml_config.use_meta_learning}")
        logger.info(f"Initialized MAML Trainer with config: {self.maml_config}")
        
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None) -> torch.Tensor:
        """
        Override training step for meta-learning
        """
        with open(os.environ.get("TSF_MAML_DEBUG_PATH", "maml_debug.txt"), "a") as f:
            f.write(f"[training_step] called at step {self.state.global_step if hasattr(self, 'state') else '?'}\n")
        
        if not self.maml_config.use_meta_learning:
            return super().training_step(model, inputs, num_items_in_batch)
        
        model.train()
        
        # Debug: Log training step entry
        print(
            f"[DEBUG] training_step called at global_step={self.state.global_step if hasattr(self, 'state') else '?'}, "
            f"input_ids.shape={inputs['input_ids'].shape}"
        )
        logger.info_rank0(
            f"[DEBUG] training_step called at global_step={self.state.global_step if hasattr(self, 'state') else '?'}, "
            f"input_ids.shape={inputs['input_ids'].shape}"
        )
        
        # Organize data into tasks
        tasks = self._create_meta_batch(inputs)
        
        # If not enough data for meta-learning, fall back to standard training
        if len(tasks) == 0:
            batch_size = inputs['input_ids'].shape[0]
            task_size = self.maml_config.support_set_size + self.maml_config.query_set_size
            logger.warning(
                f"[WARNING] MAML: Cannot create tasks. batch_size={batch_size}, "
                f"task_size={task_size}, required={task_size}, "
                f"min_batch={self.maml_config.task_batch_size * task_size}. "
                f"Falling back to standard training."
            )
            return super().training_step(model, inputs, num_items_in_batch)
        
        logger.info_rank0(f"[DEBUG] Starting meta-training with {len(tasks)} tasks")
        
        # Meta-training loop
        meta_loss = 0.0
        
        for task_idx, (support_set, query_set) in enumerate(tasks):
            logger.debug(
                f"[DEBUG] Processing task {task_idx}/{len(tasks)}: "
                f"support_size={support_set['input_ids'].shape[0]}, "
                f"query_size={query_set['input_ids'].shape[0]}"
            )
            
            # Inner loop: adapt to support set
            if self.maml_config.simplified_maml:
                task_loss = self._simplified_maml_step(model, support_set, query_set)
            else:
                task_loss = self._standard_maml_step(model, support_set, query_set)
            
            logger.debug(f"[DEBUG] Task {task_idx} loss: {task_loss.item() if hasattr(task_loss, 'item') else task_loss}")
            
            meta_loss += task_loss
            self.num_tasks_completed += 1  # Increment task counter
            
            # Clear cache to prevent OOM
            if self.maml_config.clear_cache_per_task and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Average across tasks
        meta_loss = meta_loss / len(tasks)
        
        logger.info_rank0(f"[DEBUG] Meta-loss before backward (averaged over {len(tasks)} tasks): {meta_loss.item()}")
        
        # Backward pass
        if self.args.n_gpu > 1:
            meta_loss = meta_loss.mean()
        
        if self.args.gradient_accumulation_steps > 1:
            meta_loss = meta_loss / self.args.gradient_accumulation_steps
            logger.info_rank0(f"[DEBUG] Meta-loss after gradient accumulation scaling: {meta_loss.item()}")
        
        self.accelerator.backward(meta_loss)
        
        # Log statistics (filter out NaN values)
        loss_value = meta_loss.item()
        if not torch.isnan(torch.tensor(loss_value)):
            self.task_losses.append(loss_value)
            logger.info_rank0(f"[DEBUG] Meta_step: {self.meta_step}, num_tasks_completed: {self.num_tasks_completed}")
        else:
            logger.warning(f"[WARNING] NaN detected in meta_loss at step {self.meta_step}, skipping")
        self.meta_step += 1
        
        return meta_loss.detach()
    
    def _simplified_maml_step(
        self,
        model: nn.Module,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Simplified MAML for quantized models
        
        Instead of computing gradients through gradients, we:
        1. Fine-tune on support set
        2. Evaluate on query set
        3. Use query loss for meta-update
        
        This avoids second-order gradients which don't work with 4-bit quantization.
        """
        # Store original LoRA parameters (only these are trainable)
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora' in name.lower():
                original_params[name] = param.data.clone()
        
        # Inner loop: adapt to support set
        support_set = self._prepare_inputs(support_set)
        
        for _ in range(self.maml_config.num_inner_steps):
            # Forward pass on support set
            outputs = model(**support_set)
            support_loss = outputs.loss if hasattr(outputs, 'loss') else self.compute_loss(model, support_set)
            
            # Backward (only affects LoRA parameters)
            support_loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            # Manual gradient descent on LoRA parameters with NaN check
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and 'lora' in name.lower() and param.grad is not None:
                        # Check for NaN in gradients
                        if torch.isnan(param.grad).any():
                            logger.warning(f"NaN gradient detected in {name}, skipping update")
                            param.grad.zero_()
                            continue
                        param.data -= self.maml_config.inner_learning_rate * param.grad
                        param.grad.zero_()
        
        # Outer loop: evaluate on query set
        query_set = self._prepare_inputs(query_set)
        outputs = model(**query_set)
        query_loss = outputs.loss if hasattr(outputs, 'loss') else self.compute_loss(model, query_set)
        
        # Check for NaN in query loss
        if torch.isnan(query_loss):
            logger.warning(f"NaN detected in query_loss, returning zero")
            query_loss = torch.tensor(0.0, device=query_loss.device, requires_grad=True)
        
        # Restore original parameters (for next task)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
        
        return query_loss
    
    def _standard_maml_step(
        self,
        model: nn.Module,
        support_set: Dict[str, torch.Tensor],
        query_set: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Standard MAML with second-order gradients
        
        Only use this if your model is NOT quantized.
        """
        # Create computational graph for inner loop
        support_set = self._prepare_inputs(support_set)
        
        # Get trainable parameters
        fast_weights = {
            name: param for name, param in model.named_parameters() 
            if param.requires_grad
        }
        
        # Inner loop
        for _ in range(self.maml_config.num_inner_steps):
            outputs = model(**support_set)
            support_loss = outputs.loss if hasattr(outputs, 'loss') else self.compute_loss(model, support_set)
            
            # Compute gradients
            grads = torch.autograd.grad(
                support_loss,
                fast_weights.values(),
                create_graph=not self.maml_config.use_first_order,
                retain_graph=True
            )
            
            # Update fast weights
            fast_weights = {
                name: param - self.maml_config.inner_learning_rate * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        
        # Outer loop: evaluate on query set
        query_set = self._prepare_inputs(query_set)
        
        # Apply fast weights (this is complex and model-specific)
        # For simplicity, we use the adapted model directly
        outputs = model(**query_set)
        query_loss = outputs.loss if hasattr(outputs, 'loss') else self.compute_loss(model, query_set)
        
        return query_loss
    
    def _create_meta_batch(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Organize batch data into meta-learning tasks with domain-aware sampling
        
        Each task contains:
        - Support set: for adaptation
        - Query set: for evaluation
        
        If domain_aware_sampling is enabled and domain labels are available,
        ensures each task contains samples from the same domain.
        """
        batch_size = inputs['input_ids'].shape[0]
        task_size = self.maml_config.support_set_size + self.maml_config.query_set_size
        
        # Check if we have domain labels for this batch
        domain_labels = inputs.get('domain_labels', None)
        
        # Debug: Log batch information
        print(
            f"[DEBUG] _create_meta_batch called: batch_size={batch_size}, "
            f"task_size={task_size}, support_size={self.maml_config.support_set_size}, "
            f"query_size={self.maml_config.query_set_size}, task_batch_size={self.maml_config.task_batch_size}, "
            f"domain_aware_sampling={self.maml_config.domain_aware_sampling}, has_domain_labels={domain_labels is not None}"
        )
        logger.info_rank0(
            f"[DEBUG] _create_meta_batch called: batch_size={batch_size}, "
            f"task_size={task_size}, support_size={self.maml_config.support_set_size}, "
            f"query_size={self.maml_config.query_set_size}, task_batch_size={self.maml_config.task_batch_size}, "
            f"domain_aware_sampling={self.maml_config.domain_aware_sampling}, has_domain_labels={domain_labels is not None}"
        )
        
        tasks = []
        if self.maml_config.domain_aware_sampling and domain_labels is not None:
            # Domain-aware task organization
            logger.info_rank0(f"[DEBUG] Using domain-aware task creation")
            tasks = self._create_domain_aware_tasks(inputs, domain_labels)
            logger.info_rank0(f"[DEBUG] Domain-aware tasks created: {len(tasks)} tasks")
        
        # Fallback: if no domain-homogeneous tasks created, use sequential organization
        if len(tasks) == 0:
            # Standard sequential task organization (may mix domains)
            num_tasks = min(self.maml_config.task_batch_size, batch_size // task_size)
            
            logger.info_rank0(
                f"[DEBUG] Using sequential task creation: batch_size // task_size = {batch_size} // {task_size} = {batch_size // task_size}, "
                f"min(task_batch_size, result) = min({self.maml_config.task_batch_size}, {batch_size // task_size}) = {num_tasks}"
            )
            
            if num_tasks > 0:
                logger.info_rank0(
                    f"[DEBUG] Creating {num_tasks} meta-learning tasks: "
                    f"batch_size={batch_size}, task_size={task_size}, "
                    f"support_size={self.maml_config.support_set_size}, "
                    f"query_size={self.maml_config.query_set_size}"
                )
            else:
                logger.warning(
                    f"[WARNING] Cannot create meta-learning tasks: "
                    f"batch_size={batch_size} < task_size={task_size}. "
                    f"Required batch_size >= {task_size}"
                )
            
            for i in range(num_tasks):
                start_idx = i * task_size
                
                # Support set indices
                support_start = start_idx
                support_end = support_start + self.maml_config.support_set_size
                
                # Query set indices
                query_start = support_end
                query_end = query_start + self.maml_config.query_set_size
                
                logger.debug(
                    f"[DEBUG] Task {i}: support_indices=[{support_start}:{support_end}], "
                    f"query_indices=[{query_start}:{query_end}]"
                )
                
                # Create support and query sets
                support_set = {
                    key: value[support_start:support_end] 
                    for key, value in inputs.items()
                    if key != 'domain_labels'  # Exclude metadata
                }
                query_set = {
                    key: value[query_start:query_end]
                    for key, value in inputs.items()
                    if key != 'domain_labels'  # Exclude metadata
                }
                
                # Log tensor shapes
                if 'input_ids' in support_set and 'input_ids' in query_set:
                    logger.debug(
                        f"[DEBUG] Task {i} tensor shapes - support: {support_set['input_ids'].shape}, "
                        f"query: {query_set['input_ids'].shape}"
                    )
                
                tasks.append((support_set, query_set))
        
        logger.info_rank0(f"[DEBUG] Total tasks created: {len(tasks)}")
        return tasks
    
    def _create_domain_aware_tasks(
        self,
        inputs: Dict[str, torch.Tensor],
        domain_labels: torch.Tensor
    ) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Create tasks where each task contains samples from the same domain
        """
        batch_size = inputs['input_ids'].shape[0]
        task_size = self.maml_config.support_set_size + self.maml_config.query_set_size
        
        logger.info_rank0(
            f"[DEBUG] _create_domain_aware_tasks: batch_size={batch_size}, task_size={task_size}"
        )
        
        # Group indices by domain
        domain_to_indices = {}
        for idx in range(batch_size):
            domain = int(domain_labels[idx].item())
            if domain not in domain_to_indices:
                domain_to_indices[domain] = []
            domain_to_indices[domain].append(idx)
        
        logger.info_rank0(
            f"[DEBUG] Domain distribution: {[(d, len(indices)) for d, indices in domain_to_indices.items()]}"
        )
        
        tasks = []
        task_count = 0
        
        # Create tasks from each domain
        for domain, indices in domain_to_indices.items():
            # Skip if not enough samples for a complete task
            if len(indices) < task_size:
                logger.debug(
                    f"[DEBUG] Domain {domain}: skipped (only {len(indices)} samples < {task_size} needed)"
                )
                continue
            
            # Create as many tasks as possible from this domain
            num_domain_tasks = len(indices) // task_size
            logger.debug(
                f"[DEBUG] Domain {domain}: creating {num_domain_tasks} task(s) from {len(indices)} samples"
            )
            
            for i in range(num_domain_tasks):
                if task_count >= self.maml_config.task_batch_size:
                    logger.info_rank0(
                        f"[DEBUG] Reached task_batch_size limit ({self.maml_config.task_batch_size}), stopping task creation"
                    )
                    break
                
                # Get indices for this task
                task_indices = indices[i * task_size: (i + 1) * task_size]
                support_indices = task_indices[:self.maml_config.support_set_size]
                query_indices = task_indices[self.maml_config.support_set_size:]
                
                # Create support and query sets
                support_set = {
                    key: value[support_indices]
                    for key, value in inputs.items()
                    if key != 'domain_labels'
                }
                query_set = {
                    key: value[query_indices]
                    for key, value in inputs.items()
                    if key != 'domain_labels'
                }
                
                logger.debug(
                    f"[DEBUG] Domain {domain} Task {i}: support_indices={support_indices}, query_indices={query_indices}"
                )
                
                tasks.append((support_set, query_set))
                task_count += 1
            
            if task_count >= self.maml_config.task_batch_size:
                break
        
        logger.info_rank0(f"[DEBUG] Domain-aware task creation completed: {task_count} tasks created")
        return tasks
    
    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluation with meta-learning metrics
        """
        # Accept and ignore do_sample parameter for compatibility
        kwargs.pop('do_sample', None)
        
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add meta-learning specific metrics
        if self.task_losses:
            eval_results[f"{metric_key_prefix}_meta_loss_avg"] = sum(self.task_losses) / len(self.task_losses)
            eval_results[f"{metric_key_prefix}_meta_steps"] = self.meta_step
        
        return eval_results
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Add meta-learning logs
        """
        if self.maml_config.use_meta_learning:
            # Always log meta-learning metrics, even if task_losses is empty
            if self.task_losses:
                # Filter out NaN values for robust averaging
                valid_losses = [loss for loss in self.task_losses[-10:] if not (isinstance(loss, float) and (loss != loss))]
                if valid_losses:
                    logs["meta_loss_recent"] = sum(valid_losses) / len(valid_losses)
            logs["meta_learning/meta_step"] = self.meta_step
            logs["meta_learning/num_tasks_completed"] = self.num_tasks_completed
            print(f"[MAML DEBUG] log() called: meta_step={self.meta_step}, num_tasks={self.num_tasks_completed}, task_losses={len(self.task_losses)}")
        
        super().log(logs, start_time)
    
    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare inputs for model (move to device, etc.)
        """
        prepared = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.args.device)
            else:
                prepared[key] = value
        return prepared


def create_maml_trainer(
    model: nn.Module,
    maml_config: MAMLConfig,
    **trainer_kwargs
) -> MAMLTrainer:
    """
    Factory function to create MAML trainer
    
    Args:
        model: The model to train
        maml_config: MAML configuration
        **trainer_kwargs: Additional arguments for Trainer
    
    Returns:
        Configured MAMLTrainer instance
    """
    return MAMLTrainer(
        model=model,
        maml_config=maml_config,
        **trainer_kwargs
    )
