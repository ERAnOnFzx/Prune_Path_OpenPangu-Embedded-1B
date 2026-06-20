"""
OpenPangu Mixture-of-Experts (MoE) Core Architectural Layer.
Implements Constrained K-Means Expert Clustering and Dynamic Tau-Thresholded Routing.

Compliant with openPangu University Cooperation Acceptance Guidelines.

Official Academic Citation Reference:
Chen H, Wang Y, Han K, et al. Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition.
arXiv preprint arXiv:2505.22375, 2025.
"""

import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import normalize
from k_means_constrained import KMeansConstrained


def kmeans_cluster(param_w1: torch.Tensor, group_size: int) -> np.ndarray:
    """
    Executes balance-constrained K-Means clustering over normalized linear projection weights
    to segment hidden dimensions into uniform expert groups.

    Args:
        param_w1 (torch.Tensor): Input weight tensor of shape [hidden_dimension, input_features].
        group_size (int): Expected number of neurons structured per expert group.

    Returns:
        np.ndarray: Calculated expert identity indices of shape [hidden_dimension].
    """
    # Cast projection weights to numpy CPU space for cluster fit operations
    weights_np = param_w1.detach().cpu().float().numpy()
    weights_norm = normalize(weights_np)
    
    num_clusters = param_w1.shape[0] // group_size
    
    # Enforce precise bounds to ensure uniform sizing across all designated experts
    kmeans = KMeansConstrained(
        n_clusters=num_clusters,
        size_min=group_size,
        size_max=group_size
    ).fit(weights_norm)
    
    return kmeans.labels_


def softmax_thresholding(
    logits: torch.Tensor, 
    tau: float, 
    training: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Applies dynamic thresholding routing over standard softmax token distributions.
    Selects the minimum parameter subset of activated experts whose collective probability exceeds tau.

    Args:
        logits (torch.Tensor): Unnormalized expert router scores of shape [Batch, SeqLen, NumExperts].
        tau (float): Cumulative probability density threshold bound within (0, 1].
        training (bool): If True, returns auxiliary activation probability maps for entropy computation.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - Activated sparse gate multipliers scaled via sigmoid, shape [Batch, SeqLen, NumExperts].
            - Reference base soft probability mapping (returns None if not in training state).
    """
    # 1. Quantize probabilities across the expert dimension
    y_soft = torch.softmax(logits, dim=-1)
    sorted_prob, sorted_idx = torch.sort(y_soft, dim=-1, descending=True)

    # 2. Identify the optimal activation cutoff bound matching the density threshold tau
    cum_prob = torch.cumsum(sorted_prob, dim=-1)
    mask = cum_prob < tau 
    
    select_mask = torch.zeros_like(mask, dtype=torch.bool)
    
    # Re-map the sorted true evaluation conditions back into original expert indices
    select_mask.scatter_(dim=2, index=sorted_idx, src=mask)
    
    # 3. Guardrail Fallback: Handle outlier tokens that did not trigger any threshold criteria
    no_expert_selected = select_mask.sum(dim=-1) == 0
    if no_expert_selected.any():
        max_expert_idx = sorted_idx[:, :, 0]  # Extract high probability top-1 anchor fallback
        batch_idx, seq_idx = torch.where(no_expert_selected)
        expert_idx = max_expert_idx[batch_idx, seq_idx]
        select_mask[batch_idx, seq_idx, expert_idx] = True
    
    # 4. Construct sparse masked logit matrices using numerical type limits to prevent NaN drift
    safe_inf = torch.full_like(logits, torch.finfo(logits.dtype).min)
    sparse_logits = torch.where(select_mask, logits, safe_inf)
    y_gated = torch.sigmoid(sparse_logits)
    
    if training:             
        return y_gated, y_soft
    return y_gated, None


def router_weights_init(module: nn.Module) -> None:
    """Zero-initializes linear mapping weights inside standard MoE router configurations."""
    if isinstance(module, nn.Linear):
        init.zeros_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)


class MoELayer(nn.Module):
    """
    Structured Mixture-of-Experts (MoE) Transformer block adapter.
    Handles dynamic neuron-wise parameter isolation via balance-constrained routing vectors.
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
        self.experts = experts
        self.dimension = config.hidden_size * 4
        
        # Linear projection mapping to continuous expert logit distributions
        self.mlp_router = nn.Sequential(
            nn.Linear(config.hidden_size, experts, bias=False),
        )
        self.mlp_router.apply(router_weights_init)
        
        # Enforce structural divisibility requirements
        assert self.dimension % experts == 0, (
            f"Configuration Error: Layer hidden dimension ({self.dimension}) "
            f"must be perfectly divisible by target expert count ({experts})."
        )
        self.dimension_per_expert = self.dimension // experts
        
        # Fallback to contiguous block allocation if cluster maps are not pre-defined
        if experts_id is None:
            experts_id = torch.zeros(self.dimension, dtype=torch.long)
            for i in range(experts):
                experts_id[i * self.dimension_per_expert : (i + 1) * self.dimension_per_expert] = i

        # Vectorized generation of the sparse expert partition mask matrices
        experts_id_tensor = torch.tensor(experts_id, dtype=torch.long) if not isinstance(experts_id, torch.Tensor) else experts_id.long()
        masks_accumulated = []
        for i in range(experts):
            masks_accumulated.append(experts_id_tensor == i)
            
        # Register standard structural mask matrices as static non-trainable buffers
        mask_tensor = torch.stack(masks_accumulated).to(self.mlp_router[0].weight.dtype)
        self.register_buffer('experts_masks', mask_tensor)
        
        # Runtime Telemetry metrics properties (Encapsulated execution stats)
        self.register_buffer('activated_statistics', torch.zeros(self.experts, dtype=torch.long))
        self.sparsity_tracker_active = False
        self.total_elements_processed = 0
        self.sparse_elements_dropped = 0
        
        # Default dynamic routing density constraint bound
        self.tau = getattr(train_args, "tau", 0.95)

    def reset_moe_sparsity_statistics(self) -> None:
        """Flushes active internal metrics accumulation state properties."""
        self.sparsity_tracker_active = True
        self.total_elements_processed = 0
        self.sparse_elements_dropped = 0

    def get_sparsity_statistics(self) -> Tuple[int, int]:
        """Returns the raw processed telemetry counts for evaluation logging blocks."""
        return self.total_elements_processed, self.sparse_elements_dropped

    def forward(self, hidden_states: torch.Tensor, intermediate: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes gating matrix calculations and element-wise activation filtering.

        Args:
            hidden_states (torch.Tensor): Core layer hidden activations of shape [Batch, SeqLen, HiddenSize].
            intermediate (torch.Tensor): Native FFN up-projected block activations of shape [Batch, SeqLen, Dimension].

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Sparsely filtered hidden text embeddings of shape [Batch, SeqLen, Dimension].
                - Associated raw probability vectors for cross-entropy balance loss calculation.
        """
        # 1. Compute projection score distribution maps
        score = self.mlp_router(hidden_states)  # [Batch, SeqLen, Experts]
        
        # 2. Execute sparse routing computation matching current deployment phase limits
        experts_prob, probs_for_entropy = softmax_thresholding(
            logits=score, 
            tau=self.tau, 
            training=self.training
        )
        
        if not self.training:
            # Inline performance tracking (Asynchronous non-blocking tracking pipeline)
            flat_probs = experts_prob.reshape(-1, self.experts)
            batch_stats = (flat_probs > 0).long().sum(dim=0)
            self.activated_statistics.add_(batch_stats)
            
            if self.sparsity_tracker_active:
                self.sparse_elements_dropped += (experts_prob <= 0.0).sum().detach().item()
                self.total_elements_processed += experts_prob.numel()
                
        # 3. Project token expert gate choices into concrete internal structural neural pathways
        # Matrix Multiply Shape: [Batch, SeqLen, Experts] @ [Experts, Dimension] -> [Batch, SeqLen, Dimension]
        self_moe_mask = experts_prob @ self.experts_masks

        # 4. Apply non-linear gating scaling across the primary channel pipeline
        intermediate = intermediate * self_moe_mask

        return intermediate, probs_for_entropy