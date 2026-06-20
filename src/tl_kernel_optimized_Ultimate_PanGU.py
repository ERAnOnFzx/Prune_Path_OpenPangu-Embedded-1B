"""
OpenPangu Mixture-of-Experts (MoE) Fused Triton Inference Backend Engine.
Maintains original high-performance tensor layout mapping for Ascend NPU compilation.

Compliant with openPangu University Cooperation Acceptance Guidelines (Milestones 1 & 2).

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

import torch
import torch_npu  # Strict execution order: Must be imported immediately following torch
import triton
import triton.language as tl
import torch.nn as nn

# ==================================================
# Activation
# ==================================================
@triton.jit
def fast_gelu(x):
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

@triton.jit
def fast_silu(x):
    return x * tl.sigmoid(x)

# ==================================================
# Phase 1: Prefill Router Kernel
# ==================================================
@triton.jit
def moe_router_kernel(
    router_ptr,                 # [HIDDEN_DIM, NUM_EXPERTS]
    attention_out_ptr,          # [num_tokens, HIDDEN_DIM]
    prob_ptr,                   # [N, NUM_EXPERTS] 
    weights_ptr,                # [N, NUM_EXPERTS] 
    num_tokens,
    NUM_EXPERTS: int,           # 🚨 降级为 int，防止展开
    EXPERT_POWER2: tl.constexpr,# 🚨 保持 constexpr，用于确定 Tensor 静态形状
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr, 
    router_stride_m, router_stride_n,
    attn_stride_m, attn_stride_n,
    prob_stride_m, prob_stride_n,
    weights_stride_m, weights_stride_n
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < num_tokens
    
    offs_e = tl.arange(0, EXPERT_POWER2)
    mask_e = offs_e < NUM_EXPERTS

    logits = tl.zeros((BLOCK_SIZE_M, EXPERT_POWER2), dtype=tl.float32)
    for k in range(0, HIDDEN_DIM, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < HIDDEN_DIM

        x = tl.load(
            attention_out_ptr + offs_m[:, None] * attn_stride_m + offs_k * attn_stride_n,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        w = tl.load(
            router_ptr + offs_k[:, None] * router_stride_m + offs_e * router_stride_n,
            mask=mask_k[:, None] & mask_e[None, :],
            other=0.0
        )
        logits += (tl.dot(x, w).to(tl.float32))

    # 动态屏蔽 padded 的专家
    logits = tl.where(mask_e[None, :], logits, float("-inf"))
    row_max = tl.max(logits, axis=1)
    probs = tl.exp(logits - row_max[:, None])
    probs = tl.where(mask_e[None, :], probs, 0.0)
    probs /= tl.sum(probs, axis=1)[:, None]
    
    tl.store(prob_ptr + offs_m[:, None] * prob_stride_m + offs_e * prob_stride_n, probs, mask=mask_m[:, None] & mask_e[None, :])

    sig_w = tl.sigmoid(logits)
    tl.store(weights_ptr + offs_m[:, None] * weights_stride_m + offs_e * weights_stride_n, sig_w, mask=mask_m[:, None] & mask_e[None, :])

# ==================================================
# Phase 2: Prefill Fused Tau-Thresholding Logic
# ==================================================
@triton.jit
def moe_select_expert_kernel(
    sorted_prob_ptr,            
    active_mask_ptr,            
    num_tokens,
    threshold_tau: tl.constexpr,
    NUM_EXPERTS: int,           # 🚨 降级为 int
    EXPERT_POWER2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    sorted_prob_stride_m, sorted_prob_stride_n,
    active_mask_stride_m, active_mask_stride_n
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < num_tokens
    
    offs_e = tl.arange(0, EXPERT_POWER2)
    mask_e = offs_e < NUM_EXPERTS
    
    sorted_prob = tl.load(
        sorted_prob_ptr + offs_m[:, None] * sorted_prob_stride_m + offs_e * sorted_prob_stride_n, 
        mask=mask_m[:, None] & mask_e[None, :], 
        other=0.0
    )
    
    cum_p = tl.cumsum(sorted_prob, axis=1)
    active_mask_sorted = ((cum_p < threshold_tau) | (offs_e[None, :] == 0)) & mask_e[None, :]
    
    tl.store(active_mask_ptr + offs_m[:, None] * active_mask_stride_m + offs_e * active_mask_stride_n, active_mask_sorted, mask=mask_m[:, None] & mask_e[None, :])

@triton.jit
def _sum_op(a, b):
    return a + b

# ==================================================
# Prefill Local Scan Kernel
# ==================================================
@triton.jit
def moe_local_scan_kernel(
    active_mask_ptr,            
    local_count_ptr,            
    num_tokens,
    NUM_EXPERTS: int,           # 🚨 降级为 int
    EXPERT_POWER2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, 
    active_mask_stride_m, active_mask_stride_n,
    local_count_stride_m, local_count_stride_n
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < num_tokens
    
    offs_e = tl.arange(0, EXPERT_POWER2)
    mask_e = offs_e < NUM_EXPERTS
    
    local_active_mask = tl.load(
        active_mask_ptr + offs_m[:, None] * active_mask_stride_m + offs_e * active_mask_stride_n, 
        mask=mask_m[:, None] & mask_e[None, :], 
        other=0
    )
    local_count = tl.sum(local_active_mask.to(tl.int32), axis=0)
    
    tl.store(local_count_ptr + pid * local_count_stride_m + offs_e * local_count_stride_n, local_count, mask=mask_e)

# ==================================================
# Prefill Global Scan Kernel
# ==================================================
@triton.jit
def moe_global_scan_kernel(
    local_count_ptr,
    expert_cursor_ptr,
    total_tiles_ptr,
    tiles_to_expert_ptr,
    grid_m: tl.constexpr,
    NUM_EXPERTS: int,           # 🚨 降级为 int
    EXPERT_POWER2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    local_count_stride_m, local_count_stride_n,
    expert_cursor_stride_m, 
    tiles_to_expert_stride_m,
    MAX_FILL: tl.constexpr = 512 
):
    offs_e = tl.arange(0, EXPERT_POWER2)
    mask_e = offs_e < NUM_EXPERTS
    
    global_count_result = tl.zeros((EXPERT_POWER2,), tl.int32)
    block_expert_write_indice = tl.zeros((EXPERT_POWER2,), tl.int32)
    
    # 🚨 核心优化：Triton 会将其编译为硬件循环，拒绝展开，消除寄存器溢出
    for i in tl.range(grid_m):
        local_count_result = tl.load(local_count_ptr + i * local_count_stride_m + offs_e * local_count_stride_n, mask=mask_e, other=0)
        global_count_result += local_count_result
        block_expert_write_indice = global_count_result - local_count_result
        tl.store(local_count_ptr + i * local_count_stride_m + offs_e * local_count_stride_n, block_expert_write_indice, mask=mask_e)
    
    num_tiles_per_expert = (global_count_result + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    padded_count = tl.where(mask_e, num_tiles_per_expert * BLOCK_SIZE_M, 0)
    
    inclusive_scan = tl.associative_scan(padded_count, 0, _sum_op)
    offsets = inclusive_scan - padded_count
    tl.store(expert_cursor_ptr + offs_e * expert_cursor_stride_m, offsets, mask=mask_e)
    
    tiles_write_pos = tl.associative_scan(num_tiles_per_expert, 0, _sum_op) - num_tiles_per_expert
    
    # 🚨 核心优化：转为硬件动态循环
    for i in tl.range(NUM_EXPERTS):
        mask_i = (offs_e == i)
        write_pos = tl.sum(tl.where(mask_i, tiles_write_pos, 0))
        write_num = tl.sum(tl.where(mask_i, num_tiles_per_expert, 0))
        
        fill_offsets = tl.arange(0, MAX_FILL)
        fill_mask = fill_offsets < write_num
        tl.store(tiles_to_expert_ptr + (write_pos + fill_offsets) * tiles_to_expert_stride_m, i, mask=fill_mask)
    
    if tl.max(offs_e, 0) == EXPERT_POWER2 - 1:
        total_tiles = tl.sum(num_tiles_per_expert, 0)
        tl.store(total_tiles_ptr, total_tiles)

# ==================================================
# Prefill Dispatch Kernel
# ==================================================
@triton.jit
def moe_dispatch_kernel(
    prob_sorted_index_ptr,
    expert_cursor_ptr,
    active_mask_ptr,
    sigmoid_weight_ptr,
    expert_aligned_token_index_ptr,
    expert_aligned_weight_ptr,
    block_expert_write_indices_ptr,
    num_tokens,
    NUM_EXPERTS: int,           # 🚨 降级为 int
    EXPERT_POWER2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    prob_sorted_index_stride_m, prob_sorted_index_stride_n,
    expert_cursor_stride_m,
    active_mask_stride_m, active_mask_stride_n,
    sigmoid_weight_stride_m, sigmoid_weight_stride_n,
    expert_aligned_token_index_stride_m,
    expert_aligned_weight_stride_m, 
    block_expert_write_indices_stride_m, block_expert_write_indices_stride_n,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offs_m < num_tokens
    
    offs_e = tl.arange(0, EXPERT_POWER2)
    mask_e = offs_e < NUM_EXPERTS
    
    sorted_idx = tl.load(prob_sorted_index_ptr + offs_m[:, None] * prob_sorted_index_stride_m + offs_e * prob_sorted_index_stride_n, mask=mask_m[:, None] & mask_e[None, :], other=0)
    active_mask_sorted = tl.load(active_mask_ptr + offs_m[:, None] * active_mask_stride_m + offs_e * active_mask_stride_n, mask=mask_m[:, None] & mask_e[None, :], other=0)
    sigmoid_weight = tl.load(sigmoid_weight_ptr + offs_m[:, None] * sigmoid_weight_stride_m + offs_e * sigmoid_weight_stride_n, mask=mask_m[:, None] & mask_e[None, :], other=0.0)
    
    # 🚨 核心优化：转为硬件动态循环，中断寄存器溢出，极大提升吞吐
    for j in tl.range(NUM_EXPERTS):
        mask_j_2d = (sorted_idx == j) & active_mask_sorted 
        mask_j = tl.sum(mask_j_2d.to(tl.int32), axis=1) > 0     
        num_tokens_for_j = tl.sum(mask_j)
        
        if num_tokens_for_j > 0:
            base_pos = tl.load(expert_cursor_ptr + j * expert_cursor_stride_m)
            offset_pos = tl.load(block_expert_write_indices_ptr + pid * block_expert_write_indices_stride_m + j * block_expert_write_indices_stride_n)
            block_start_pos = base_pos + offset_pos
            mask_j_int = mask_j.to(tl.int32)
            local_offsets = tl.cumsum(mask_j_int, axis=0) - 1
            write_indices = block_start_pos + local_offsets
            
            sig_w = tl.sum(tl.where(offs_e == j, sigmoid_weight, 0.0), axis=1)
            
            tl.store(expert_aligned_token_index_ptr + write_indices * expert_aligned_token_index_stride_m, offs_m, mask=mask_j)
            tl.store(expert_aligned_weight_ptr + write_indices * expert_aligned_weight_stride_m, sig_w, mask=mask_j)

# ==========================================
# Prefill Phase: Fused SwiGLU Expert Compute
# ==========================================
@triton.jit
def moe_fused_expert_kernel(
    X_ptr, W_gate_ptr, W_up_ptr, W_down_ptr, Out_ptr,
    expert_aligned_weight_ptr,
    expert_aligned_token_index_ptr,
    tiles_to_expert_ptr,
    H: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K_GEMM: tl.constexpr,  # 🚨 接收优化后的小 Block K (如 64)
    X_stride_m, X_stride_n,
    W_gate_stride_m, W_gate_stride_n,
    W_up_stride_m, W_up_stride_n,
    W_down_stride_m, W_down_stride_n,
    Out_stride_m, Out_stride_n,
    expert_aligned_weight_stride_m,
    expert_aligned_token_index_stride_m,
    tiles_to_expert_stride_m
):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    expert_id = tl.load(tiles_to_expert_ptr + pid * tiles_to_expert_stride_m)
    token_indices = tl.load(expert_aligned_token_index_ptr + pid * BLOCK_SIZE_M * expert_aligned_token_index_stride_m + offs_m * expert_aligned_token_index_stride_m)

    acc_gate = tl.zeros((BLOCK_SIZE_M, GROUP_SIZE), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, GROUP_SIZE), dtype=tl.float32)
    offs_g = tl.arange(0, GROUP_SIZE)
    expert_offset = expert_id * GROUP_SIZE

    # 🚨 [优化] 使用更细粒度的 BLOCK_K_GEMM 激活 Tensor Core 的流水线并行
    for k in range(0, H, BLOCK_K_GEMM):
        offs_k = k + tl.arange(0, BLOCK_K_GEMM)
        mask_k = offs_k < H
        X = tl.load(X_ptr + token_indices[:, None] * X_stride_m + offs_k[None, :] * X_stride_n, mask=mask_k[None, :], other=0.0)
        
        W_gate = tl.load(W_gate_ptr + offs_k[:, None] * W_gate_stride_m + (offs_g[None, :] + expert_offset) * W_gate_stride_n, mask=mask_k[:, None], other=0.0)
        W_up = tl.load(W_up_ptr + offs_k[:, None] * W_up_stride_m + (offs_g[None, :] + expert_offset) * W_up_stride_n, mask=mask_k[:, None], other=0.0)
        
        # 🚨 [优化] 移除了对 W 冗余的 .to(tl.bfloat16)，因为内存中本身就是 BF16
        acc_gate += tl.dot(X, W_gate).to(tl.float32)
        acc_up += tl.dot(X, W_up).to(tl.float32)

    acc_gate = fast_silu(acc_gate) * acc_up

    act_w = tl.load(expert_aligned_weight_ptr + pid * BLOCK_SIZE_M * expert_aligned_weight_stride_m + offs_m * expert_aligned_weight_stride_m)
    acc_gate *= act_w[:, None]
    acc_gate = acc_gate.to(tl.bfloat16)

    for k in range(0, H, BLOCK_K_GEMM):
        offs_k_out = k + tl.arange(0, BLOCK_K_GEMM)
        mask_k_out = offs_k_out < H
        W_down = tl.load(W_down_ptr + (expert_offset + offs_g[:, None]) * W_down_stride_m + offs_k_out[None, :] * W_down_stride_n, mask=mask_k_out[None, :], other=0.0)
        
        acc_down = tl.dot(acc_gate, W_down)
        
        # 🚨 [优化] 直接向 BF16 的 Out_ptr 执行 atomic_add，减少显存读写带宽占用
        tl.atomic_add(Out_ptr + token_indices[:, None] * Out_stride_m + offs_k_out[None, :] * Out_stride_n, acc_down.to(tl.bfloat16), mask=mask_k_out[None, :])


# ====================================================================================================================================================== #
#                                                               Specialized For Decoding Phase                                                              #
# ====================================================================================================================================================== #

@triton.jit
def moe_router_select_kernel_decode(
    router_ptr, attention_out_ptr, active_idx_ptr, active_weight_ptr, num_active_experts_ptr,
    threshold_tau: tl.constexpr, 
    NUM_EXPERTS: int, 
    EXPERT_POWER2: tl.constexpr, 
    HIDDEN_DIM: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    router_stride_m, router_stride_n, attn_stride_n, active_idx_stride_n, active_weight_stride_n
):
    offs_e = tl.arange(0, EXPERT_POWER2)
    mask_e = offs_e < NUM_EXPERTS

    logits = tl.zeros((EXPERT_POWER2,), dtype=tl.float32)
    for k in range(0, HIDDEN_DIM, BLOCK_SIZE_K):
        offs_k = k + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < HIDDEN_DIM

        x = tl.load(attention_out_ptr + offs_k * attn_stride_n, mask=mask_k, other=0.0)
        w = tl.load(router_ptr + offs_k[:, None] * router_stride_m + offs_e[None, :] * router_stride_n, mask=mask_k[:, None] & mask_e[None, :], other=0.0)
        logits += tl.sum(x[:, None] * w, axis=0).to(tl.float32)

    logits = tl.where(mask_e, logits, float("-inf"))
    row_max = tl.max(logits, axis=0)
    probs = tl.exp(logits - row_max)
    probs = tl.where(mask_e, probs, 0.0)
    probs = probs / tl.sum(probs, axis=0)
    sig_w = tl.sigmoid(logits)

    remaining = probs
    cumulative = tl.full((), 0.0, dtype=tl.float32)
    num_active = tl.full((), 0, dtype=tl.int32)

    for rank in tl.range(NUM_EXPERTS):
        max_prob = tl.max(remaining, axis=0)
        expert_id = tl.min(tl.where(remaining == max_prob, offs_e, NUM_EXPERTS), axis=0)
        active = ((cumulative + max_prob) < threshold_tau) | (rank == 0)

        selected_weight = tl.sum(tl.where(offs_e == expert_id, sig_w, 0.0), axis=0)
        tl.store(active_idx_ptr + rank * active_idx_stride_n, expert_id, mask=active)
        tl.store(active_weight_ptr + rank * active_weight_stride_n, selected_weight, mask=active)
        num_active += active.to(tl.int32)

        remaining = tl.where(offs_e == expert_id, -1.0, remaining)
        cumulative += max_prob

    tl.store(num_active_experts_ptr, num_active)

# ==================================================
# 🚨 [核心升级] Decode Phase 2: Fully Fused SwiGLU (Gate -> Up -> Down)
# ==================================================
@triton.jit
def moe_decode_swiglu_fused_kernel(
    X_ptr, W_gate_ptr, W_up_ptr, W_down_ptr, Out_ptr,
    active_idx_ptr, active_weight_ptr, num_active_experts_ptr,
    H: tl.constexpr, GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_H: tl.constexpr,
    x_stride_n, 
    w_gate_stride_m, w_gate_stride_n, 
    w_up_stride_m, w_up_stride_n, 
    w_down_stride_m, w_down_stride_n, 
    out_stride_n,
    active_idx_stride_n, active_weight_stride_n
):
    pid_expert = tl.program_id(0)
    pid_g = tl.program_id(1)  # 沿着 Group Size 的切片维度调度

    num_active = tl.load(num_active_experts_ptr)
    if pid_expert >= num_active:
        return

    expert_id = tl.load(active_idx_ptr + pid_expert * active_idx_stride_n)
    act_w = tl.load(active_weight_ptr + pid_expert * active_weight_stride_n)

    expert_offset = expert_id * GROUP_SIZE
    offs_g = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
    mask_g = offs_g < GROUP_SIZE

    # ----------------------------------------------------
    # Sub-Phase 1: Local Gate & Up Projections
    # 计算一小块 (BLOCK_G) 大小的中间激活向量并保留在寄存器中
    # ----------------------------------------------------
    acc_gate = tl.zeros((BLOCK_G,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_G,), dtype=tl.float32)
    
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H
        
        x = tl.load(X_ptr + offs_k * x_stride_n, mask=mask_k, other=0.0)
        
        w_gate = tl.load(
            W_gate_ptr + offs_k[:, None] * w_gate_stride_m + (expert_offset + offs_g[None, :]) * w_gate_stride_n,
            mask=mask_k[:, None] & mask_g[None, :],
            other=0.0
        )
        w_up = tl.load(
            W_up_ptr + offs_k[:, None] * w_up_stride_m + (expert_offset + offs_g[None, :]) * w_up_stride_n,
            mask=mask_k[:, None] & mask_g[None, :],
            other=0.0
        )
        
        acc_gate += tl.sum(x[:, None] * w_gate, axis=0).to(tl.float32)
        acc_up += tl.sum(x[:, None] * w_up, axis=0).to(tl.float32)

    # 寄存器内极速完成 SiLU 和路由权重相乘
    act = fast_silu(acc_gate) * act_w * acc_up
    act = act.to(tl.bfloat16)

    # ----------------------------------------------------
    # Sub-Phase 2: Down Projection & Global Atomic Reduce
    # 复用刚才计算出的 act，直接通过 W_down 映射回 H 维度
    # ----------------------------------------------------
    for h in range(0, H, BLOCK_H):
        offs_h = h + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        
        # 加载对应的 W_down 块
        w_down = tl.load(
            W_down_ptr + (expert_offset + offs_g[:, None]) * w_down_stride_m + offs_h[None, :] * w_down_stride_n,
            mask=mask_g[:, None] & mask_h[None, :],
            other=0.0
        )
        
        # 矩阵乘向量：act [BLOCK_G] @ w_down [BLOCK_G, BLOCK_H]
        partial_out = tl.sum(act[:, None] * w_down, axis=0)
        
        # 并发 L2 原子累加（针对 bfloat16 非常高效，远胜于内存落盘）
        tl.atomic_add(Out_ptr + offs_h * out_stride_n, partial_out, mask=mask_h)


# ==================================================
# Main MoE Module
# ==================================================
class TauSparseMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts: int, group_size: int, tau: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.group_size = group_size
        self.tau = tau

        self.EXPERTS_POWER2 = triton.next_power_of_2(self.num_experts)

        self.BLOCK_K_GEMM = 64          
        self.BLOCK_T = 16               
        self.BLOCK_H_EW = 1024          
        
        self.register_buffer("W_gate", torch.empty(hidden_size, intermediate_size, dtype=torch.bfloat16))
        self.register_buffer("W_up", torch.empty(hidden_size, intermediate_size, dtype=torch.bfloat16))
        self.register_buffer("W_down", torch.empty(intermediate_size, hidden_size, dtype=torch.bfloat16))
        self.register_buffer("router", torch.empty(hidden_size, num_experts, dtype=torch.bfloat16))
        
        self.register_buffer("decode_active_idx", torch.empty(num_experts, dtype=torch.int32), persistent=False)
        self.register_buffer("decode_active_weight", torch.empty(num_experts, dtype=torch.float32), persistent=False)
        self.register_buffer("decode_num_active", torch.empty(1, dtype=torch.int32), persistent=False)
        self.register_buffer("decode_out", torch.empty(1, hidden_size, dtype=torch.bfloat16), persistent=False)
        

    def forward_decode(self, x_flat, B, L, H, device):

        # 🚨 atomic_add 要求在每一轮前向传播之前必须清零输出容器！
        self.decode_out.zero_()

        # =================================================================
        # Phase 1: 路由计算 (Top-Tau Select)
        # =================================================================
        moe_router_select_kernel_decode[(1,)](
            self.router, x_flat, self.decode_active_idx, self.decode_active_weight, self.decode_num_active,
            self.tau, self.num_experts, self.EXPERTS_POWER2, 
            H, self.BLOCK_K_GEMM,
            self.router.stride(0), self.router.stride(1), 
            x_flat.stride(1),
            self.decode_active_idx.stride(0), self.decode_active_weight.stride(0)
        )


        # =================================================================
        # Phase 2: 一次性全融合的 SwiGLU 计算 (Gate & Up -> Down)
        # =================================================================
        grid_fused = (self.num_experts,)
        
        moe_decode_swiglu_fused_kernel[grid_fused](
            x_flat, self.W_gate, self.W_up, self.W_down, self.decode_out,
            self.decode_active_idx, self.decode_active_weight, self.decode_num_active,
            H, self.group_size,
            self.BLOCK_K_GEMM, self.BLOCK_K_GEMM, self.BLOCK_K_GEMM, # K, G, H 的分块统一复用 GEMM 尺寸
            x_flat.stride(1), 
            self.W_gate.stride(0), self.W_gate.stride(1), 
            self.W_up.stride(0), self.W_up.stride(1), 
            self.W_down.stride(0), self.W_down.stride(1), 
            self.decode_out.stride(1),
            self.decode_active_idx.stride(0), self.decode_active_weight.stride(0)
        )

        return self.decode_out.view(B, L, H)
    
    def forward_prefill(self, N, x_flat, B, L, H, device):
        grid_m = triton.cdiv(N, self.BLOCK_T)

        # 🚨 [优化] Python 侧显存生命周期与分配优化
        soft_prob = torch.empty(N, self.num_experts, device=device, dtype=torch.float32)
        sig_weight = torch.empty(N, self.num_experts, device=device, dtype=torch.float32)
        active_mask = torch.empty(N, self.num_experts, device=device, dtype=torch.bool)
        
        # 🚨 [优化] 移除 .clone()，直接申请空张量
        select_mask = torch.empty(N, self.num_experts, device=device, dtype=torch.bool)
        
        expert_offset = torch.empty(self.num_experts, device=device, dtype=torch.int32)
        tiles_to_expert_map = torch.empty(N * self.num_experts, device=device, dtype=torch.int32)
        total_tiles = torch.empty(1, device=device, dtype=torch.int32)
        
        # 🚨 [优化] 输出张量直接采用 bfloat16，显存与带宽减半
        out_flat = torch.zeros(N, H, device=device, dtype=torch.bfloat16)

        moe_router_kernel[grid_m,](
            self.router, x_flat, soft_prob, sig_weight,
            N, self.num_experts, self.EXPERTS_POWER2, self.hidden_size, self.BLOCK_T, self.BLOCK_K_GEMM,
            self.router.stride(0), self.router.stride(1), x_flat.stride(0), x_flat.stride(1), soft_prob.stride(0), soft_prob.stride(1), sig_weight.stride(0), sig_weight.stride(1)
        )
        
        # 原生 sort 依然是瓶颈，但在纯 Python 侧暂无更好替代，保持原样
        sorted_prob, sorted_idx = torch.sort(soft_prob, dim=-1, descending=True)          
        
        moe_select_expert_kernel[grid_m,](
            sorted_prob, active_mask, N, self.tau, self.num_experts, self.EXPERTS_POWER2, self.BLOCK_T, 
            sorted_prob.stride(0), sorted_prob.stride(1), active_mask.stride(0), active_mask.stride(1)
        )
        
        select_mask.scatter_(dim=1, index=sorted_idx, src=active_mask)
        
        # 🚨 [优化] 干净地申请小块 Int 张量，而非从 Float 张量非法转换
        # grid_local_count = torch.empty((grid_m, self.num_experts), device=device, dtype=torch.int32)
        grid_local_count = soft_prob[:grid_m].to(torch.int32)
        
        moe_local_scan_kernel[grid_m,](
            select_mask, grid_local_count, N, self.num_experts, self.EXPERTS_POWER2, self.BLOCK_T,
            select_mask.stride(0), select_mask.stride(1), grid_local_count.stride(0), grid_local_count.stride(1)
        )
    
        moe_global_scan_kernel[1,](
            grid_local_count, expert_offset, total_tiles, tiles_to_expert_map,
            grid_m, self.num_experts, self.EXPERTS_POWER2, self.BLOCK_T,
            grid_local_count.stride(0), grid_local_count.stride(1), expert_offset.stride(0), tiles_to_expert_map.stride(0), MAX_FILL=512
        )
        
        TotalTiles = total_tiles.item()
        # if TotalTiles > 0:
        expert_aligned_token_idx = torch.zeros(TotalTiles * self.BLOCK_T, device=device, dtype=torch.int32)
        expert_aligned_weight = torch.zeros(TotalTiles * self.BLOCK_T, device=device, dtype=torch.float32)
        
        moe_dispatch_kernel[grid_m,](
            sorted_idx, expert_offset, active_mask, sig_weight, expert_aligned_token_idx, expert_aligned_weight, grid_local_count,
            N, self.num_experts, self.EXPERTS_POWER2, self.BLOCK_T, 
            sorted_idx.stride(0), sorted_idx.stride(1), expert_offset.stride(0), active_mask.stride(0), active_mask.stride(1), sig_weight.stride(0), sig_weight.stride(1),
            expert_aligned_token_idx.stride(0), expert_aligned_weight.stride(0), grid_local_count.stride(0), grid_local_count.stride(1),
        )
        
        # 🚨 [优化] 传入 BLOCK_K_GEMM 供矩阵乘使用
        moe_fused_expert_kernel[TotalTiles,](
            x_flat, self.W_gate, self.W_up, self.W_down, out_flat, expert_aligned_weight, expert_aligned_token_idx, tiles_to_expert_map,
            H, self.group_size, self.BLOCK_T, self.BLOCK_K_GEMM,
            x_flat.stride(0), x_flat.stride(1),
            self.W_gate.stride(0), self.W_gate.stride(1),
            self.W_up.stride(0), self.W_up.stride(1),
            self.W_down.stride(0), self.W_down.stride(1),
            out_flat.stride(0), out_flat.stride(1), expert_aligned_weight.stride(0), expert_aligned_token_idx.stride(0), tiles_to_expert_map.stride(0)
        )
        
        return out_flat.view(B, L, H)

    def forward(self, x):
        B, L, H = x.shape
        N = B * L
        device = x.device
        x_flat = x.reshape(N, H).to(torch.bfloat16)
        
        if N == 1:
            return self.forward_decode(x_flat, B, L, H, device)
        else:
            return self.forward_prefill(N, x_flat, B, L, H, device)