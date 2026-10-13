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

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig


class TemporalAwareDropout(nn.Module):
    """
    Temporal-aware dropout layer for time series prediction tasks.
    
    This dropout layer considers the temporal structure of the input sequence,
    applying different dropout strategies based on time position and sequence characteristics.
    """
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        temporal_decay_factor: float = 0.8,
        structured_dropout_prob: float = 0.1,
        adaptive_dropout: bool = False,
        sequence_length: int = 96,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.temporal_decay_factor = temporal_decay_factor
        self.structured_dropout_prob = structured_dropout_prob
        self.adaptive_dropout = adaptive_dropout
        self.sequence_length = sequence_length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal-aware dropout to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)
            
        Returns:
            Tensor with temporal-aware dropout applied
        """
        if not self.training or self.dropout_rate == 0.0:
            return x
            
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. Time-weighted dropout: Recent timesteps have lower dropout probability
        time_weights = self._compute_time_weights(seq_len)
        
        # 2. Adaptive dropout based on sequence variance
        if self.adaptive_dropout:
            adaptive_factor = self._compute_adaptive_factor(x)
            time_weights = time_weights * adaptive_factor
            
        # 3. Apply structured dropout with probability
        if torch.rand(1).item() < self.structured_dropout_prob:
            dropout_mask = self._create_structured_dropout_mask(x, time_weights)
        else:
            dropout_mask = self._create_temporal_dropout_mask(x, time_weights)
            
        return x * dropout_mask
    
    def _compute_time_weights(self, seq_len: int) -> torch.Tensor:
        """
        Compute time-based weights for dropout.
        More recent timesteps (higher indices) get lower dropout rates.
        """
        # Create exponential decay weights
        positions = torch.arange(seq_len, dtype=torch.float32, device=next(self.parameters()).device)
        # Normalize positions to [0, 1]
        normalized_positions = positions / (seq_len - 1) if seq_len > 1 else torch.zeros_like(positions)
        
        # Apply exponential decay: recent timesteps (higher positions) have lower dropout
        decay_weights = torch.exp(-self.temporal_decay_factor * (1 - normalized_positions))
        
        # Convert to dropout probabilities
        dropout_probs = self.dropout_rate * (1 - decay_weights)
        dropout_probs = torch.clamp(dropout_probs, 0.0, 0.9)  # Ensure reasonable bounds
        
        return 1 - dropout_probs  # Convert to keep probabilities
    
    def _compute_adaptive_factor(self, x: torch.Tensor) -> float:
        """
        Compute adaptive factor based on input sequence variance.
        Higher variance sequences get higher dropout to prevent overfitting.
        """
        # Compute variance across the sequence dimension
        sequence_var = torch.var(x, dim=1).mean().item()
        
        # Normalize variance and convert to adaptive factor
        # Higher variance -> higher dropout
        adaptive_factor = 1.0 + 0.5 * torch.tanh(torch.tensor(sequence_var - 1.0)).item()
        return torch.clamp(torch.tensor(adaptive_factor), 0.5, 2.0).item()
    
    def _create_temporal_dropout_mask(self, x: torch.Tensor, time_weights: torch.Tensor) -> torch.Tensor:
        """
        Create dropout mask with temporal awareness.
        """
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # Expand time weights to match input dimensions
        time_weights_expanded = time_weights.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, hidden_dim)
        
        # Generate random mask
        random_mask = torch.rand_like(x, device=device)
        
        # Apply temporal weights
        dropout_mask = (random_mask < time_weights_expanded).float()
        
        # Scale to maintain expected value
        dropout_mask = dropout_mask / (time_weights_expanded + 1e-8)
        
        return dropout_mask
    
    def _create_structured_dropout_mask(self, x: torch.Tensor, time_weights: torch.Tensor) -> torch.Tensor:
        """
        Create structured dropout mask that drops consecutive time segments.
        For wind speed data with constant segments, we adjust the segment length
        to preserve important temporal patterns while still providing regularization.
        """
        batch_size, seq_len, hidden_dim = x.shape
        device = x.device
        
        # Initialize mask with all ones
        dropout_mask = torch.ones_like(x, device=device)
        
        # For wind speed data, we use variable segment lengths based on data characteristics:
        # 1. Short segments (1-5% of sequence) for preserving local patterns
        # 2. Medium segments (5-15% of sequence) for moderate regularization
        # 3. Longer segments (15-25% of sequence) for stronger regularization
        
        # Determine segment length based on random selection to provide varied regularization
        rand_choice = torch.rand(1).item()
        if rand_choice < 0.6:  # 60% chance for short segments
            min_segment_len = max(1, seq_len // 100)  # ~1% of sequence
            max_segment_len = max(min_segment_len, seq_len // 20)  # ~5% of sequence
        elif rand_choice < 0.9:  # 30% chance for medium segments
            min_segment_len = max(1, seq_len // 20)  # ~5% of sequence
            max_segment_len = max(min_segment_len, seq_len // 10)  # ~10% of sequence
        else:  # 10% chance for longer segments
            min_segment_len = max(1, seq_len // 10)  # ~10% of sequence
            max_segment_len = max(min_segment_len, seq_len // 4)  # ~25% of sequence
            
        segment_len = torch.randint(min_segment_len, max_segment_len + 1, (1,)).item()
        
        # Protect the most recent timesteps (last 10% of sequence) more for wind speed prediction
        # as recent data is more important for forecasting
        protected_len = max(1, seq_len // 10)
        max_start_pos = seq_len - segment_len - protected_len
        
        if max_start_pos > 0:
            # Randomly select start position for dropout segment, but prefer earlier positions
            # for wind speed data since earlier patterns are important for learning temporal dependencies
            position_distribution = torch.softmax(torch.linspace(1.0, 0.5, max_start_pos), dim=0)
            start_pos = torch.multinomial(position_distribution, 1).item()
            end_pos = start_pos + segment_len
            
            # Apply structured dropout to the selected segment
            dropout_mask[:, start_pos:end_pos, :] = 0.0
            
            # Compensate for the dropped values by scaling remaining values
            remaining_ratio = (seq_len - segment_len) / seq_len
            dropout_mask = dropout_mask / remaining_ratio
            
            # Additional adjustment for wind speed data:
            # Apply gradual dropout within the segment to preserve some information
            # This helps maintain the temporal continuity that is important in wind speed data
            fade_len = min(segment_len // 2, 3)  # Fade in/out over up to 3 timesteps or half segment
            if fade_len > 0:
                for i in range(fade_len):
                    fade_factor = i / fade_len
                    if start_pos - fade_len + i >= 0:
                        dropout_mask[:, start_pos - fade_len + i, :] *= (1 - fade_factor)
                    if end_pos + i < seq_len:
                        dropout_mask[:, end_pos + i, :] *= fade_factor
        
        return dropout_mask


class TemporalLoraConfig(LoraConfig):
    """
    Extended LoRA configuration with temporal-aware dropout parameters.
    """
    
    def __init__(
        self,
        temporal_dropout: bool = False,
        temporal_decay_factor: float = 0.8,
        structured_dropout_prob: float = 0.1,
        adaptive_dropout: bool = False,
        sequence_length: int = 96,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.temporal_dropout = temporal_dropout
        self.temporal_decay_factor = temporal_decay_factor
        self.structured_dropout_prob = structured_dropout_prob
        self.adaptive_dropout = adaptive_dropout
        self.sequence_length = sequence_length


def create_temporal_lora_config(finetuning_args) -> LoraConfig:
    """
    Create LoRA configuration with temporal-aware dropout if enabled.
    
    Args:
        finetuning_args: Finetuning arguments containing LoRA parameters
        
    Returns:
        LoRA configuration (temporal-aware if enabled, standard otherwise)
    """
    base_config = {
        "r": finetuning_args.lora_rank,
        "lora_alpha": finetuning_args.lora_alpha,
        "lora_dropout": finetuning_args.lora_dropout,
        "use_rslora": finetuning_args.use_rslora,
        "use_dora": finetuning_args.use_dora,
    }
    
    print(f"Temporal dropout enabled: {finetuning_args.lora_temporal_dropout}")  # 调试信息
    if finetuning_args.lora_temporal_dropout:
        print("Using TemporalLoraConfig")  # 调试信息
        print(f"Temporal decay factor: {finetuning_args.lora_temporal_decay_factor}")  # 调试信息
        print(f"Structured dropout prob: {finetuning_args.lora_structured_dropout_prob}")  # 调试信息
        print(f"Adaptive dropout: {finetuning_args.lora_adaptive_dropout}")  # 调试信息
        return TemporalLoraConfig(
            temporal_dropout=True,
            temporal_decay_factor=finetuning_args.lora_temporal_decay_factor,
            structured_dropout_prob=finetuning_args.lora_structured_dropout_prob,
            adaptive_dropout=finetuning_args.lora_adaptive_dropout,
            sequence_length=finetuning_args.lora_sequence_length,
            **base_config
        )
    else:
        print("Using standard LoraConfig")  # 调试信息
        return LoraConfig(**base_config)