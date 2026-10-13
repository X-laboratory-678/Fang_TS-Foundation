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
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lora import LoraLayer
from peft.utils import transpose

from .temporal_lora import TemporalAwareDropout


class TemporalLoraLinear(nn.Module, LoraLayer):
    """
    LoRA linear layer with temporal-aware dropout for time series prediction.
    
    This layer extends the standard LoRA implementation by incorporating
    temporal-aware dropout that considers the sequential nature of time series data.
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        # Temporal-aware parameters
        temporal_dropout: bool = False,
        temporal_decay_factor: float = 0.8,
        structured_dropout_prob: float = 0.1,
        adaptive_dropout: bool = False,
        sequence_length: int = 96,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        
        # Initialize LoRA parameters
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        
        # Initialize temporal-aware dropout
        self.temporal_dropout_enabled = temporal_dropout
        if temporal_dropout:
            self.temporal_dropout = TemporalAwareDropout(
                dropout_rate=lora_dropout,
                temporal_decay_factor=temporal_decay_factor,
                structured_dropout_prob=structured_dropout_prob,
                adaptive_dropout=adaptive_dropout,
                sequence_length=sequence_length,
            )
        else:
            self.temporal_dropout = None
            
        self.is_target_conv_1d_layer = is_target_conv_1d_layer
    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):
        """Update LoRA layer parameters."""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
            
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
            
        self.lora_dropout[adapter_name] = lora_dropout_layer
        
        # Actual trainable parameters
        if self.base_layer.__class__.__name__ == "Linear":
            in_features, out_features = self.base_layer.in_features, self.base_layer.out_features
        elif self.base_layer.__class__.__name__ == "Conv1d":
            in_features, out_features = self.base_layer.in_channels, self.base_layer.out_channels
        else:
            raise ValueError(f"Unsupported layer type: {self.base_layer.__class__.__name__}")
            
        self.lora_A[adapter_name] = nn.Linear(in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, out_features, bias=False)
        
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r
            
        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
            
        # DoRA implementation would go here if needed
        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False
            
        self.set_adapter(adapter_name)
    
    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        """Reset LoRA parameters."""
        if init_lora_weights is False:
            return
            
        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # Initialize A with kaiming uniform and B with zeros
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
    
    def loftq_init(self, adapter_name):
        """LoftQ initialization (placeholder)."""
        # This would implement LoftQ initialization if needed
        pass
    
    def dora_init(self, adapter_name):
        """DoRA initialization (placeholder)."""
        # This would implement DoRA initialization if needed
        pass
    
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass with temporal-aware dropout.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_dim)
            
        Returns:
            Output tensor with LoRA adaptation and temporal-aware dropout applied
        """
        previous_dtype = x.dtype
        
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                    
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                
                x = x.to(lora_A.weight.dtype)
                
                # Apply temporal-aware dropout if enabled
                if self.temporal_dropout_enabled and self.temporal_dropout is not None:
                    x_dropped = self.temporal_dropout(x)
                else:
                    x_dropped = dropout(x)
                
                # Standard LoRA computation
                if not self.use_dora[active_adapter]:
                    lora_output = lora_B(lora_A(x_dropped)) * scaling
                else:
                    # DoRA computation would go here
                    lora_output = lora_B(lora_A(x_dropped)) * scaling
                
                lora_output = lora_output.to(torch_result_dtype)
                result = result + lora_output
                
        result = result.to(previous_dtype)
        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        if self.temporal_dropout_enabled:
            rep += f", temporal_dropout=True"
        return rep


def replace_lora_layers_with_temporal(model, finetuning_args):
    """
    Replace standard LoRA layers with temporal-aware LoRA layers.
    
    Args:
        model: The model to modify
        finetuning_args: Finetuning arguments containing temporal parameters
        
    Returns:
        Modified model with temporal-aware LoRA layers
    """
    if not finetuning_args.lora_temporal_dropout:
        return model
        
    # This function would recursively replace LoRA layers
    # Implementation would depend on the specific PEFT structure
    # For now, this is a placeholder that would be integrated with PEFT
    
    return model