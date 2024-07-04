from mamba_ssm import Mamba2
import torch
from typing import Any, Dict, Optional


class BasicMamba2Block(Mamba2):
    def __init__(
        self,
        d_model,
        **kwargs
    ):
        """
        Args:
            d_model (int): Dimensionality of the model.
            **kwargs: Additional keyword arguments including:
                - d_state=128,
                - d_conv=4,
                - conv_init=None,
                - expand=2,
                - headdim=64,
                - d_ssm=None,  # If not None, only apply SSM on this many dimensions, the rest uses gated MLP
                - ngroups=1,
                - A_init_range=(1, 16),
                - D_has_hdim=False,
                - rmsnorm=True,
                - norm_before_gate=False,
                - dt_min=0.001,
                - dt_max=0.1,
                - dt_init_floor=1e-4,
                - dt_limit=(0.0, float("inf")),
                - bias=False,
                - conv_bias=True,
                - chunk_size=256,
                - use_mem_eff_path=True,
                - layer_idx=None,
                - process_group=None,
                - sequence_parallel=True,
                - device=None,
                - dtype=None
        """
        super().__init__(d_model, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ):
        return super().forward(hidden_states)