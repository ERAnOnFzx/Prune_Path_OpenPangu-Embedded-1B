"""
OpenPangu-MoE (Origin Variant) Performance Profiling & Evaluation Pipeline
Designed for Automated Weight Verification and Benchmark Testing on Ascend NPU.

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

import os
import gc
import torch
import torch_npu  # Strict execution order: Must be imported immediately following torch
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from dataclasses import dataclass

# Ensure exact mapping to the finetuned MoE structural repository layer
from openpangu_moe_origin import CustomOpenPangu 

# ==============================================================================
# 1. Global Performance Testing Constants & Hyperparameters
# ==============================================================================
MAX_LENGTH = 512
STRIDE = 256
BATCH_SIZE = 1
DEFAULT_MAX_NEW_TOKENS = 128
SUM_TOK = "<SUMMARY>"

@dataclass
class CustomTrainingArgs:
    """Standardized hyperparameter alignment layer for MoE compatibility."""
    stage1_epoch: int = 1
    stage2_epoch: int = 1
    batch: int = BATCH_SIZE 
    gradient_accumulation_steps: int = 1
    stage1_lr: float = 1e-4
    stage2_lr: float = 1e-4
    decay: float = 0.0          
    moe_lr: float = 1e-3        
    moe_decay: float = 1e-2     
    gamma: float = 3e-4         
    lb_gamma: float = 1e-3      
    group_size: int = 32
    dataset: str = "xsum"       
    tau: float = 0.70
    n_inner: int = None
    stage1_tau: float = 1.05
    stage2_tau: float = 0.50

def parse_args():
    """Parses environment execution configs and local weight checkpoint paths."""
    parser = argparse.ArgumentParser(description="Performance Profiling Suite for openPangu MoE Models")
    parser.add_argument("--checkpoint", type=str, default="../weights/stage2_tau70_best.pt", 
                        help="Path to the best fine-tuned MoE model weights checkpoint")
    parser.add_argument("--num_samples", type=int, default=2, 
                        help="Number of operational sample streams to loop through for profiling statistics.")
    parser.add_argument("--tau", type=float, default=0.7, help="Setting the tau threshold.")
    return parser.parse_args()


# ==============================================================================
# 2. Dataset Pipeline & Structured Prompt Synthesis
# ==============================================================================
def prepare_summarization_prompt(dataset, tokenizer, max_input_length=384, index=0):
    """
    Extracts pure textual sequences and appends explicit summary triggers.
    Optimized to eliminate data preprocessing overhead from latency calculations.
    """
    example = dataset["validation"][index]
    document = example["document"]
    
    text = f"{document}{SUM_TOK}"
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    sum_token_id = tokenizer.convert_tokens_to_ids(SUM_TOK)
    if input_ids[0, -1].item() != sum_token_id:
        input_ids[0, -1] = sum_token_id
        
    summary = example["summary"]
    gt_len = tokenizer(summary, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }, gt_len


# ==============================================================================
# 3. Telemetry Modules & Performance Measurement Engines
# ==============================================================================
@torch.no_grad()
def sanity_check_kv_cache(model, prompt):
    """Verifies state registration and tensor structural integrity of the MoE KV-Cache."""
    model.eval()

    with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
        out = model(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            use_cache=True,
        )

    print("\n================== [Telemetry: KV-Cache Integrity Check] ==================")
    print(f"[*] Layer State Allocation Active : {out.past_key_values is not None}")
    if out.past_key_values is not None:
        if hasattr(out.past_key_values, "get_seq_length"):
            cache_len = out.past_key_values.get_seq_length()
        elif isinstance(out.past_key_values, tuple) and len(out.past_key_values) > 0:
            cache_len = out.past_key_values[0][0].shape[-2]
        else:
            cache_len = "Unknown Data Structure"
            
        prompt_len = prompt["input_ids"].shape[1]
        print(f"[*] Registered Cache Context Steps: {cache_len}")
        print(f"[*] Priming Sequence Length       : {prompt_len}")
    print("===========================================================================\n")


@torch.no_grad()
def benchmark_peak_vram(model, prompt, tokenizer, max_new_tokens):
    """Measures maximum high-bandwidth memory (HBM) usage during autoregressive execution."""
    model.eval()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

    with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
        _ = model.model.generate(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    torch.npu.synchronize()
    peak_mem = torch.npu.max_memory_allocated() / (1024 ** 2)
    return peak_mem


@torch.no_grad()
def benchmark_e2e_generate_latency(model, prompt, tokenizer, max_new_tokens, warmup_iters=10, latency_iters=50):
    """Calculates steady-state generation latency and overall end-to-end processing throughput."""
    model.eval()

    with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
        for _ in range(warmup_iters):
            _ = model.model.generate(
                input_ids=prompt["input_ids"],
                attention_mask=prompt["attention_mask"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
        for _ in range(latency_iters):
            _ = model.model.generate(
                input_ids=prompt["input_ids"],
                attention_mask=prompt["attention_mask"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    end_event.record()
    torch.npu.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / latency_iters
    throughput = (BATCH_SIZE * max_new_tokens) / (avg_latency_ms / 1000.0)
    return avg_latency_ms, throughput


def fast_clone_kv_cache(kv_cache):
    """High-speed key-value state deep copying module to eliminate iteration leakage."""
    if kv_cache is None:
        return None
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(kv_cache, DynamicCache):
            new_cache = DynamicCache()
            for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
                new_cache.update(k.clone(), v.clone(), layer_idx=len(new_cache.key_cache))
            return new_cache
    except ImportError:
        pass
    if isinstance(kv_cache, tuple):
        return tuple(tuple(t.clone() for t in layer) for layer in kv_cache)
    return kv_cache


@torch.no_grad()
def benchmark_manual_decode_only(model, prompt, tokenizer, max_new_tokens, profile_iters=5):
    """Isolates and evaluates the token generation (decode) loop by stripping prefill bias."""
    model.eval()

    input_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]

    with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    base_past_key_values = outputs.past_key_values
    first_next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

    total_decode_ms = 0.0
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    for it in range(profile_iters):
        current_kv = fast_clone_kv_cache(base_past_key_values)
        current_input = first_next_token.clone()
        curr_attention_mask = attention_mask.clone() if attention_mask is not None else None

        torch.npu.synchronize()
        start_event.record()

        for _step in range(max_new_tokens - 1):
            if curr_attention_mask is not None:
                ones = torch.ones(
                    (curr_attention_mask.shape[0], 1),
                    dtype=curr_attention_mask.dtype,
                    device=curr_attention_mask.device,
                )
                curr_attention_mask = torch.cat([curr_attention_mask, ones], dim=-1)

            with torch.amp.autocast(device_type="npu", dtype=torch.bfloat16):
                out = model(
                    input_ids=current_input,
                    past_key_values=current_kv,
                    attention_mask=curr_attention_mask,
                    use_cache=True,
                )

            current_input = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
            current_kv = out.past_key_values

        end_event.record()
        torch.npu.synchronize()
        total_decode_ms += start_event.elapsed_time(end_event)

    avg_decode_ms = total_decode_ms / profile_iters
    decode_steps = max_new_tokens - 1
    throughput = (BATCH_SIZE * decode_steps) / (avg_decode_ms / 1000.0) if avg_decode_ms > 0 else 0

    return avg_decode_ms, throughput


# ==============================================================================
# 4. Main Instrumentation Orchestrator Pipeline
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    
    # --------------------------------------------------------------------------
    # Ascend Hardware Platform Environmental Setup (Milestone 1 Core Acceptance Logs)
    # --------------------------------------------------------------------------
    if torch.npu.is_available():
        device = torch.device("npu:0")
        torch.npu.set_device(device)
        
        # Avoid unexpected dynamic kernel re-compilations disrupting precise timing steps
        try:
            torch.npu.set_compile_mode(jit_compile=False)
        except Exception:
            pass
            
        # Standardize hardware layout memory shapes by locking dynamic internal translations
        try:
            torch.npu.config.allow_internal_format = False
        except Exception:
            pass
            
        # Dynamically retrieve hardware device name to pass compliance checks (Avoid static strings) 
        device_name = torch.npu.get_device_name(device)
        print(f"[INFO] 硬件环境就绪：当前硬件为 【{device_name}】，本程序正基于昇腾运行。")
    else:
        device = torch.device("cpu")
        device_name = "CPU Backend"
        print(f"[WARNING] 硬件环境警告：未检测到昇腾计算硬件，回退至 【{device_name}】。测速指标将失效。")

    train_args = CustomTrainingArgs()
    if args.tau:
        train_args.tau = args.tau

    # --------------------------------------------------------------------------
    # Component Tokenizer & OpenPangu Model Config Mapping
    # --------------------------------------------------------------------------
    model_name = "FreedomIntelligence/openPangu-Embedded-1B"
    print(f"[INFO] 模型架构配置：正在加载模型为openPangu 系列开源模型，当前实例变量路径为: '{model_name}'")
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config._attn_implementation = "eager" 

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"additional_special_tokens": [SUM_TOK]})

    print("[INFO] Fetching evaluation dataset split (XSum: Validation Partition)...")
    raw_dataset = load_dataset("EdinburghNLP/xsum")

    print("[INFO] Initializing CustomOpenPangu MoE structural topology...")
    model = CustomOpenPangu(model_name_or_path=model_name, config=config, train_args=train_args)
    model.model.resize_token_embeddings(len(tokenizer))
    
    # Initialize structural expert routing gate masks
    model.custom_post_init() 
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[INFO] Restoring fine-tuned MoE parameters from local checkpoint path: '{args.checkpoint}'")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)

    model.to(device, dtype=torch.bfloat16)

    # Prime execution environment via verification run using reference sequence #1000
    print("[INFO] Simulating verification context to anchor memory layout boundaries...")
    test_prompt, _ = prepare_summarization_prompt(raw_dataset, tokenizer, index=1000)
    test_prompt = {k: v.to(device) for k, v in test_prompt.items()}
    sanity_check_kv_cache(model, test_prompt)

    # Accumulation arrays for target metrics
    peak_vrams = []
    e2e_latencies = []
    e2e_throughputs = []
    decode_latencies = []
    decode_throughputs = []

    print(f"\n================== [Initiating Multi-Sample Profiler Task Array ({args.num_samples} streams)] ==================")
    
    for i in range(args.num_samples):
        current_index = 1000 + i
        print(f"[PROFILER] Stream Index Process: {i+1}/{args.num_samples} (Source Row Pointer: {current_index})")
        
        prompt, gt_len = prepare_summarization_prompt(
            raw_dataset, tokenizer, max_input_length=MAX_LENGTH, index=current_index
        )
        prompt = {k: v.to(device) for k, v in prompt.items()}
        current_max_new_tokens = gt_len
        
        # Phase 1: Dynamic HBM High Watermark Registration
        vram = benchmark_peak_vram(model, prompt, tokenizer, current_max_new_tokens)
        peak_vrams.append(vram)
        
        # Phase 2: Complete Sequence Time-to-Last-Token End-to-End Analysis (High-Iteration Iteration)
        e2e_lat, e2e_thr = benchmark_e2e_generate_latency(
            model=model, prompt=prompt, tokenizer=tokenizer, 
            max_new_tokens=current_max_new_tokens, warmup_iters=10, latency_iters=50
        )
        e2e_latencies.append(e2e_lat)
        e2e_throughputs.append(e2e_thr)
        
        # Phase 3: Token Generation Inter-Token Isolated Decode Analysis (High-Iteration Iteration)
        dec_lat, dec_thr = benchmark_manual_decode_only(
            model=model, prompt=prompt, tokenizer=tokenizer,
            max_new_tokens=current_max_new_tokens, profile_iters=5
        )
        decode_latencies.append(dec_lat)
        decode_throughputs.append(dec_thr)

        # Strict memory wall: Purge context residue to lock benchmark sample purity
        del prompt
        gc.collect()
        torch.npu.empty_cache()

    print("\n================== [Benchmark Task Array Concluded. Serializing Summary Statistics] ==================")
    
    avg_vram = np.mean(peak_vrams)
    avg_e2e_lat = np.mean(e2e_latencies)
    avg_e2e_thr = np.mean(e2e_throughputs)
    avg_dec_lat = np.mean(decode_latencies)
    avg_dec_thr = np.mean(decode_throughputs)

    print(f"[SUCCESS] Profiling iterations completed : {args.num_samples}")
    print(f"[SUCCESS] Mean High-Watermark HBM Allocation : {avg_vram:.2f} MB")
    print(f"[SUCCESS] Normalized System End-to-End Latency : {avg_e2e_lat:.2f} ms")
    print(f"[SUCCESS] Steady-state Processing Output Flow  : {avg_e2e_thr:.2f} tokens/s")
    print(f"[SUCCESS] Isolated Decoupled Loop Generation Lat: {avg_dec_lat:.2f} ms")
    print(f"[SUCCESS] Isolated Decoupled Loop Output Speed  : {avg_dec_thr:.2f} tokens/s")

    print("\n[Copy-friendly result block]")
    print(f"origin_avg_peak_vram_mb = {avg_vram:.4f}")
    print(f"origin_avg_e2e_latency_ms = {avg_e2e_lat:.4f}")
    print(f"origin_avg_e2e_throughput = {avg_e2e_thr:.4f}")
    print(f"origin_avg_decode_latency_ms = {avg_dec_lat:.4f}")
    print(f"origin_avg_decode_throughput = {avg_dec_thr:.4f}")

    print("====================================================================================================\n")
