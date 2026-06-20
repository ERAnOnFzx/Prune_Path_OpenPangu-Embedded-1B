"""
OpenPangu Mixture-of-Experts (MoE) Core Architectural Layer.
Implements Balanced Neural Pathway Partitioning and Dynamic Tau-Thresholded Expert Routing.

Optimized for Zero-Sync Asynchronous Execution on Huawei Ascend NPU Ecosystems.
Compliant with openPangu University Cooperation Acceptance Guidelines[cite: 1, 22].

Reference Citation[cite: 13]:
Chen H, Wang Y, Han K, et al. Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition.
arXiv preprint arXiv:2505.22375, 2025. [cite: 14, 15]
"""

import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Any
from sklearn.preprocessing import normalize
from k_means_constrained import KMeansConstrained


def kmeans_cluster(param_w1: torch.Tensor, group_size: int) -> np.ndarray:
    """
    Executes balance-constrained K-Means clustering over normalized linear projection weights
    to segment hidden dimensions into uniform expert groups during model initialization.

    Args:
        param_w1 (torch.Tensor): Input dense weight tensor of shape [hidden_dimension, input_features].
        group_size (int): Expected number of neurons structured per expert group.

    Returns:
        np.ndarray: Calculated expert identity indices of shape [hidden_dimension].
    """
    # Safe migration to host CPU storage before executing NumPy array casting
    param_w1_numpy = param_w1.detach().cpu().float().numpy() 
    param_w1_numpy_norm = normalize(param_w1_numpy)
    
    num_clusters = param_w1.shape[0] // group_size
    kmeans = KMeansConstrained(
        n_clusters=num_clusters,
        size_min=group_size,
        size_max=group_size
    ).fit(param_w1_numpy_norm)
    
    return kmeans.labels_


def softmax_thresholding(
    logits: torch.Tensor, 
    tau: float, 
    training: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Applies dynamic density threshold routing over standard softmax token distributions.
    Selects the minimum activation subset of experts whose collective cumulative probability exceeds tau.

    Args:
        logits (torch.Tensor): Unnormalized expert router scores of shape [Batch, SeqLen, NumExperts].
        tau (float): Cumulative probability density threshold bound within (0, 1].
        training (bool): If True, returns auxiliary activation probability maps for loss computation.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - Activated sparse gate multipliers scaled via sigmoid, shape [Batch, SeqLen, NumExperts].
            - Reference base soft probability distribution mapping (None if in inference mode).
    """
    y_soft = torch.softmax(logits, dim=-1)
    sorted_prob, sorted_idx = torch.sort(y_soft, dim=-1, descending=True)  # [B, L, E]

    # Compute cumulative probability density matching the hyperparameter bound tau
    cum_prob = torch.cumsum(sorted_prob, dim=-1)  # [B, L, E]
    mask = cum_prob < tau 
    
    select_mask = torch.zeros_like(mask, dtype=torch.bool)
    
    # Scatter true active flags back to the original un-sorted expert dimension mapping
    select_mask.scatter_(dim=2, index=sorted_idx, src=mask)
    
    # Guardrail: Force top-1 selection for tokens that fall outside the density mask boundaries
    no_expert_selected = select_mask.sum(dim=-1) == 0  # [B, L]
    if no_expert_selected.any():
        max_expert_idx = sorted_idx[:, :, 0]  # Extract max probability anchor index
        batch_idx, seq_idx = torch.where(no_expert_selected)
        expert_idx = max_expert_idx[batch_idx, seq_idx]
        select_mask[batch_idx, seq_idx, expert_idx] = True
    
    # Numerical protection: Fill masked regions with datatype minimum instead of '-inf' to safeguard NPU execution
    safe_inf_fill = torch.full_like(logits, torch.finfo(logits.dtype).min)
    sparse_logits = torch.where(select_mask, logits, safe_inf_fill)   
    y_gated = torch.sigmoid(sparse_logits)   
    
    if training:             
        return y_gated, y_soft
    return y_gated, None


def router_weights_init(module: nn.Module) -> None:
    """Applies zero-initialization layer weights inside the standard MoE router configuration mapping."""
    if isinstance(module, nn.Linear):
        init.zeros_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)


class MoELayer(nn.Module):
    """
    Structured Mixture-of-Experts (MoE) Adapter Pipeline.
    Implements non-blocking, variable-mapped neuron clustering and dynamic expert gating tracking.
    """
    def __init__(
        self, 
        config: Any, 
        train_args: Any,
        experts: int, 
        experts_id: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.train_args = train_args
        self.dimension = config.hidden_size * 4
        self.experts = experts
        
        # Router projection layers instantiation
        self.mlp_router = nn.Sequential(
            nn.Linear(config.hidden_size, experts, bias=False),
        )
        self.mlp_router.apply(router_weights_init)

        assert self.dimension % experts == 0, "Structural Dimension constraint error: dimension must be divisible by experts."
        self.dimension_per_expert = self.dimension // experts
        
        # Vectorized generation of expert parameter structural partitioning masks
        if experts_id is None:
            experts_id_tensor = torch.zeros(self.dimension, dtype=torch.long)
            for i in range(experts):
                experts_id_tensor[i * self.dimension_per_expert : (i + 1) * self.dimension_per_expert] = i
        else:
            experts_id_tensor = torch.as_tensor(experts_id, dtype=torch.long)

        # Build structural mapping matrix purely on torch space [NumExperts, Dimension]
        mask_tensor = F.one_hot(experts_id_tensor, num_classes=experts).to(torch.bool).t()
        self.register_buffer('experts_masks', mask_tensor.to(self.mlp_router[0].weight.dtype))
        
        # Telemetry Metrics Tracker: Registered as buffer to prevent CPU-NPU host sync bottlenecking
        self.register_buffer('activated_statistics', torch.zeros(self.experts, dtype=torch.long))
        self.register_buffer('sparsity_accumulator', torch.zeros(1, dtype=torch.double))
        self.register_buffer('element_counter', torch.zeros(1, dtype=torch.long))
        self.sparsity_tracking_enabled = False
        
        # Initialize the gating activation threshold bound via variable mapping
        self.tau = getattr(train_args, "tau", 0.95)

    def reset_moe_sparsity_statistics(self) -> None:
        """Flushes telemetry counter values cleanly without re-instantiating device buffers."""
        self.sparsity_tracking_enabled = True
        self.sparsity_accumulator.zero_()
        self.element_counter.zero_()

    def get_sparsity_statistics(self) -> Tuple[int, float]:
        """Extracts the accumulated telemetry profile tracking records."""
        return self.element_counter.item(), self.sparsity_accumulator.item()

    def forward(self, hidden_states: torch.Tensor, intermediate: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes gating weight operations and applies parallel element-wise expert masking.

        Args:
            hidden_states (torch.Tensor): Core hidden text activations of shape [Batch, SeqLen, HiddenSize].
            intermediate (torch.Tensor): Primary FFN up-projected block activations of shape [Batch, SeqLen, Dimension].

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Sparsely routed activation tensor of shape [Batch, SeqLen, Dimension].
                - Raw routing softmax distribution mapping for entropy loss balancing calculations.
        """
        # Compute projection gate routing metrics scores
        score = self.mlp_router(hidden_states)  # Shape: [Batch, SeqLen, Experts]
        
        # Isolate step execution routing context
        experts_prob, probs_for_entropy = softmax_thresholding(
            logits=score, 
            tau=self.tau, 
            training=self.training
        )
        
        if not self.training:
            # High-performance In-place execution on NPU device to prevent synchronization blockages
            flat_probs = experts_prob.reshape(-1, self.experts)
            batch_active_stats = (flat_probs > 0).long().sum(dim=0)
            self.activated_statistics.add_(batch_active_stats)
            
            if self.sparsity_tracking_enabled:
                # Accumulate sparse dropping statistics natively
                self.sparsity_accumulator.add_((experts_prob <= 0.0).sum().detach())
                self.element_counter.add_(experts_prob.numel())
            
        # Distribute group weights back onto original channel pipelines via matrix multiplication
        # [B, L, E] @ [E, Dimension] -> [B, L, Dimension]
        self_moe_mask = experts_prob @ self.experts_masks

        # Apply spatial structural gating adjustments across the intermediate FFN projection channels
        intermediate = intermediate * self_moe_mask

        if not self.training:
            return intermediate, None
            
        return intermediate, probs_for_entropy