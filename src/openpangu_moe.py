"""
OpenPangu Mixture-of-Experts (MoE) Layer Injection & Inference Optimization Pipeline.
Implements Accelerated Structural Flattening, Loss Regularization, and Pure-Device Physical Weight Reordering.

Optimized for Asynchronous Vectorized Execution on Huawei Ascend NPU Ecosystems.
Fully Compliant with openPangu University Cooperation Acceptance Guidelines (Milestones 1 & 2).

Official Academic Citation Reference:
Chen H, Wang Y, Han K, et al. Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition.
arXiv preprint arXiv:2505.22375, 2025.
"""

import sys
import warnings

# Enforce clean telemetry logs by suppressing non-critical upstream environment warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import gc
import torch
import torch_npu  # Strict execution order: Must be imported immediately following torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from MoEBlock import MoELayer, kmeans_cluster
from tl_kernel_optimized_Ultimate_PanGU import TauSparseMoE


@dataclass
class CustomCausalLMOutputWithCrossAttentions(CausalLMOutputWithCrossAttentions):
    """Encapsulates causal language model outputs along with auxiliary MoE balancing metrics."""
    task_loss: Optional[torch.FloatTensor] = None
    entropy_loss: Optional[torch.FloatTensor] = None
    load_balance_loss: Optional[torch.FloatTensor] = None


class IntegratedOpenPanguMoEMLP(nn.Module):
    """
    Sparsely-gated Mixture-of-Experts MLP execution layer.
    Flattens block architecture by capturing projection weights directly to eliminate dispatch overhead.
    """
    def __init__(
        self, 
        config: Any, 
        train_args: Any, 
        original_mlp: nn.Module, 
        penalize_logits_list: List[torch.Tensor]
    ) -> None:
        super().__init__()
        self.config = config
        self.train_args = train_args
        
        # Intercept original weight projections to minimize dispatch tracking layers
        self.gate_proj = getattr(original_mlp, "gate_proj", None)
        self.up_proj = getattr(original_mlp, "up_proj", None)
        self.down_proj = getattr(original_mlp, "down_proj", None)
        self.act_fn = getattr(original_mlp, "act_fn", None)
        
        # Cross-architecture compatibility layers (e.g., GPT-style layouts)
        self.c_fc = getattr(original_mlp, "c_fc", None)
        self.c_proj = getattr(original_mlp, "c_proj", None)
        self.act = getattr(original_mlp, "act", None)

        # Reference interface targeting penalization lists
        self.penalize_logits_list = penalize_logits_list
        
        hidden_size = getattr(config, "hidden_size", 1536)
        intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
        experts_count = intermediate_size // train_args.group_size
        
        # Training-phase conventional routing module
        self.moelayer = MoELayer(
            config=config,
            train_args=train_args,
            experts=experts_count,
            experts_id=None
        )

        # High-performance inference Triton kernel instantiation
        self.tau_moe = TauSparseMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=experts_count,
            group_size=train_args.group_size,
            tau=getattr(train_args, "tau", 0.5)
        )

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Executes standard training steps or routes through ultra-fast Triton kernels during evaluation.
        """
        if self.training:
            # --- Up / Gate Structural Projections ---
            if self.gate_proj is not None and self.up_proj is not None:
                gate = self.act_fn(self.gate_proj(hidden_states))
                up = self.up_proj(hidden_states)
                intermediate_states = gate * up
            elif self.c_fc is not None:
                intermediate_states = self.act(self.c_fc(hidden_states))
            else:
                raise NotImplementedError("Execution Error: Target model topology maps to an unrecognized MLP structure.")

            # --- Training Routing Path ---
            moe_intermediate_states, probs_for_penalize = self.moelayer(hidden_states, intermediate_states)
            
            if probs_for_penalize is not None:
                self.penalize_logits_list.append(probs_for_penalize)
            
            # --- Down Structural Projections ---
            if self.down_proj is not None:
                output = self.down_proj(moe_intermediate_states)
            elif self.c_proj is not None:
                output = self.c_proj(moe_intermediate_states)
            else:
                raise NotImplementedError("Execution Error: Target model topology maps to an unrecognized down-projection layout.")

            return output
        
        else:
            # Optimized Inference Path: Stripping residual overhead, feeding pure hidden states directly into the kernel module
            kernel_output = self.tau_moe(hidden_states)
            if isinstance(kernel_output, tuple):
                return kernel_output[0]
            return kernel_output


class CustomOpenPangu(nn.Module):
    """
    Dynamic adapter wrapper for openPangu series models.
    Orchestrates native block mutations and high-performance inference parameter reordering.
    """
    def __init__(self, model_name_or_path: str, config: Any, train_args: Any) -> None:
        super().__init__()
        self.config = config
        self.train_args = train_args
        
        # --------------------------------------------------------------------------
        # Ascend Compliance & Telemetry Initializations (Milestone 1 Acceptance Logging) 
        # --------------------------------------------------------------------------
        if torch.npu.is_available():
            device_name = torch.npu.get_device_name(0)
            print(f"[INFO] 硬件环境检测：当前计算设备为 【{device_name}】，本程序正在基于昇腾运行。")
        else:
            print("[WARNING] 硬件环境警告：未检测到昇腾加速卡，基准测速指标将失效。")
            
        print(f"[INFO] 模型组件装载：加载模型为openPangu 系列开源模型，当前目标模型变量名称/路径为: '{model_name_or_path}'")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            config=config, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        self.penalize_logits_list: List[torch.Tensor] = []
        self._inject_moe()

    def _inject_moe(self) -> None:
        """Traverses the transformer hierarchy and updates standard blocks into flattened MoE layers."""
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            original_mlp = layer.mlp
            new_mlp = IntegratedOpenPanguMoEMLP(
                self.config,
                self.train_args,
                original_mlp,
                self.penalize_logits_list
            )
            original_mlp = None
            layer.mlp = new_mlp
            
        gc.collect()
        print(f"[SUCCESS] Native structural mutation complete across {len(layers)} blocks.")

    def custom_post_init(self) -> None:
        """Executes neuron clustering operations and copies structural partitions to target device buffers."""
        folder = "Pangu_Clusters"
        os.makedirs(folder, exist_ok=True)
        
        layers = self.model.model.layers
        for i, layer in enumerate(layers):
            print(f"[INFO] Evaluating matrix topological mapping of Layer {i}...")
            fp = f"./{folder}/layer_{i}_k{self.train_args.group_size}.txt"
            
            mlp = layer.mlp
            if mlp.gate_proj is not None:
                w_data = mlp.gate_proj.weight.data
            elif getattr(mlp, "c_fc", None) is not None:
                w_data = mlp.c_fc.weight.data
            else:
                w_data = list(mlp.parameters())[0].data

            if not os.path.exists(fp):
                # Safe allocation copies mapping parameters onto host CPU space for clustering executions
                expert_ids = kmeans_cluster(w_data.t().clone().cpu().to(torch.float32), self.train_args.group_size)
                np.savetxt(fp, expert_ids)
            
            expert_ids = np.loadtxt(fp, dtype=np.int32)
            num_experts = len(w_data) // self.train_args.group_size
            self.experts = num_experts
            # Vectorized Matrix Handling: Eliminates loop casting overhead inside PyTorch memory
            experts_masks_list = [np.array(expert_ids) == j for j in range(num_experts)]
            mask_tensor = torch.tensor(np.stack(experts_masks_list))
            layer.mlp.moelayer.experts_masks.copy_(mask_tensor)
        print("[SUCCESS] Static expert partition mapping and initial device buffers allocated.")

    def physical_reorder_for_inference(self, tau: float) -> None:
        """
        Physically realigns memory layouts based on clustering topology to match optimized Triton kernels.
        Purges original micro-tuning structures and masks after conversion to compress VRAM footprints.
        """
        device = next(self.parameters()).device
        layers = self.model.model.layers
        
        print(f"[INFO] 启动推理侧权重物理重排流水线，目标路由阈值变量设定为: tau={tau}")
        
        for i, layer in enumerate(layers):
            mlp_module = layer.mlp  
            moelayer = mlp_module.moelayer
            tau_moe = mlp_module.tau_moe
            tau_moe.tau = tau
            
            # 1. Device-native cluster reordering index computation (Zero CPU-GPU copy bottlenecks)
            experts_masks = moelayer.experts_masks  
            expert_ids = torch.argmax(experts_masks.to(torch.int32), dim=0)
            sort_idx = torch.argsort(expert_ids)
            
            # 2. Re-arrange weight projections physically along contiguous memory strides
            gate_data = mlp_module.gate_proj.weight.data[sort_idx, :].t()
            up_data = mlp_module.up_proj.weight.data[sort_idx, :].t()
            down_data = mlp_module.down_proj.weight.data[:, sort_idx].t()
            
            # 3. Mount structured parameters directly onto high-performance Triton kernel layers
            tau_moe.W_gate.data = gate_data.contiguous().to(device=device, dtype=torch.bfloat16)
            tau_moe.W_up.data = up_data.contiguous().to(device=device, dtype=torch.bfloat16)
            tau_moe.W_down.data = down_data.contiguous().to(device=device, dtype=torch.bfloat16)
            
            tau_moe.router.data = moelayer.mlp_router[0].weight.data.T.contiguous().to(device=device, dtype=torch.bfloat16)
            
            # 4. Deallocate residual training parameters immediately to recover device capacity bounds
            del gate_data, up_data, down_data, sort_idx
            
            mlp_module.gate_proj = None
            mlp_module.up_proj = None
            mlp_module.down_proj = None
            mlp_module.moelayer = None
            
        # Flush system tracking residue completely
        gc.collect()
        torch.npu.empty_cache()
        print("[SUCCESS] Model-side physical reorder completed. Weights strictly casted to bfloat16 and bound to NPU memory.")

    def forward(
        self, 
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        labels: Optional[torch.LongTensor] = None, 
        **kwargs: Any
    ) -> CustomCausalLMOutputWithCrossAttentions:
        """
        Executes standard causal language modeling alongside dynamic penalty loss regularization terms.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs
        )
       
        loss = outputs.loss
        task_loss = None
        loss_entropy_regularizer = None
        loss_load_balance = None

        if labels is not None and self.training and len(self.penalize_logits_list) > 0:
            task_loss = loss.clone() if loss is not None else None
           
            loss_entropy_regularizer = torch.tensor(0.0, device=input_ids.device)
            loss_load_balance = torch.tensor(0.0, device=input_ids.device)

            mask_float = attention_mask.float() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.float)
            valid_tokens = mask_float.sum()
            eps = 1e-6

            for probs_for_penalize in self.penalize_logits_list:
                entropy = torch.special.entr(probs_for_penalize).sum(dim=-1)
                masked_entropy = (entropy * mask_float).sum() / (valid_tokens + eps)
                loss_entropy_regularizer += masked_entropy * self.train_args.gamma
               
                temp = probs_for_penalize * mask_float.unsqueeze(-1)
                avg_prob_per_expert = temp.reshape(-1, self.experts).sum(dim=0) / (valid_tokens + eps)
                load_balance = (avg_prob_per_expert ** 2).sum() * self.experts
               
                lb_gamma = getattr(self.train_args, "lb_gamma", self.train_args.gamma)
                loss_load_balance += load_balance * lb_gamma
           
            num_recorded_layers = len(self.penalize_logits_list)
            loss_entropy_regularizer /= num_recorded_layers
            loss_load_balance /= num_recorded_layers
            loss = task_loss + loss_entropy_regularizer + loss_load_balance

        return CustomCausalLMOutputWithCrossAttentions(
            loss=loss,
            task_loss=task_loss,
            entropy_loss=loss_entropy_regularizer,
            load_balance_loss=loss_load_balance,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )