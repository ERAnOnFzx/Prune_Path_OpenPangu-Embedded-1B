"""
OpenPangu-MoE Dense Baseline Performance Profiling & Evaluation Pipeline
Designed for Automated Benchmarking on Huawei Ascend NPU Platforms.

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
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
import torch.nn as nn

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
    tau: float = 0.50
    n_inner: int = None
    stage1_tau: float = 1.05
    stage2_tau: float = 0.50

def parse_args():
    """Parses environment execution configs and data precision arguments."""
    parser = argparse.ArgumentParser(description="Performance Profiling Suite for openPangu Models")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to dense baseline checkpoint (Optional)")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of operational sample streams to profile.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="NPU execution precision data type (bf16 recommended for Ascend 910B).")
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
# 3. Environment Seeding & High-Performance Cache Cloning
# ==============================================================================
def seed_all(seed: int):
    """Guarantees deterministic execution behavior across multi-backend setups."""
    import random
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is out of bounds [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)

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
def sanity_check_kv_cache(model, prompt):
    """Verifies state registration and tensor structural integrity of the KV-Cache."""
    model.eval()
    with torch.amp.autocast("npu", dtype=torch.bfloat16):
        out = model(input_ids=prompt["input_ids"], attention_mask=prompt["attention_mask"], use_cache=True)

    print("\n================== [Telemetry: KV-Cache Integrity Check] ==================")
    print(f"[*] Layer State Allocation Active : {out.past_key_values is not None}")
    
    if out.past_key_values is not None:
        cache_len = out.past_key_values.get_seq_length() if hasattr(out.past_key_values, "get_seq_length") else out.past_key_values[0][0].shape[-2]
        prompt_len = prompt["input_ids"].shape[1]
        print(f"[*] Registered Cache Context Steps: {cache_len}")
        print(f"[*] Priming Sequence Sequence Length : {prompt_len}")
    print("===========================================================================\n")


# ==============================================================================
# 4. Dense Native MLP Architectural Profiler
# ==============================================================================
class MLPTimingRecorder:
    """Monitors and aggregates asynchronous operational costs inside feed-forward sub-networks."""
    def __init__(self):
        self.enabled = False
        self.records = []

    def reset(self):
        self.records = []

    def add(self, start_event, end_event, layer_id, phase, num_tokens):
        self.records.append({
            "start": start_event,
            "end": end_event,
            "layer_id": layer_id,
            "phase": phase,
            "num_tokens": num_tokens,
        })

    def summarize(self, num_generated_tokens, num_layers, profiled_iters, profiled_e2e_ms=None):
        torch.npu.synchronize()
        summary = defaultdict(float)
        count = defaultdict(int)

        for rec in self.records:
            elapsed = rec["start"].elapsed_time(rec["end"])
            phase = rec["phase"]

            summary[f"{phase}_mlp_ms"] += elapsed
            summary["total_mlp_ms"] += elapsed
            count[f"{phase}_calls"] += 1
            count["total_calls"] += 1

        total_mlp_per_iter = summary["total_mlp_ms"] / max(profiled_iters, 1)
        prefill_mlp_per_iter = summary["prefill_mlp_ms"] / max(profiled_iters, 1)
        decode_mlp_per_iter = summary["decode_mlp_ms"] / max(profiled_iters, 1)
        
        observed_decode_steps = count["decode_calls"] / max(profiled_iters * num_layers, 1)
        decode_mlp_per_generated_token = decode_mlp_per_iter / max(num_generated_tokens, 1)
        decode_mlp_per_layer_generated_token = decode_mlp_per_generated_token / max(num_layers, 1)
        decode_mlp_per_decode_step = decode_mlp_per_iter / max(observed_decode_steps, 1)
        decode_mlp_per_layer_decode_step = decode_mlp_per_decode_step / max(num_layers, 1)

        print("\n================== [Architectural Profile: Dense Native MLP] ==================")
        print(f"[*] Total Execution Dispatches     : {count['total_calls']}")
        print(f"[*] Prefill Sub-phase Invocations  : {count['prefill_calls']}")
        print(f"[*] Autoregressive Decode Triggers : {count['decode_calls']}")
        print(f"[*] Normalized Multi-step MLP Cost : {total_mlp_per_iter:.4f} ms")
        print(f"[*] Isolated Context Prefill Latency: {prefill_mlp_per_iter:.4f} ms")
        print(f"[*] Isolated Token Generation Cost  : {decode_mlp_per_iter:.4f} ms")
        print("===============================================================================\n")

        return {
            "total_mlp_per_iter_ms": total_mlp_per_iter,
            "prefill_mlp_per_iter_ms": prefill_mlp_per_iter,
            "decode_mlp_per_iter_ms": decode_mlp_per_iter,
            "observed_decode_steps": observed_decode_steps,
            "decode_mlp_per_generated_token_ms": decode_mlp_per_generated_token,
            "decode_mlp_per_layer_generated_token_ms": decode_mlp_per_layer_generated_token,
            "decode_mlp_per_decode_step_ms": decode_mlp_per_decode_step,
            "decode_mlp_per_layer_decode_step_ms": decode_mlp_per_layer_decode_step,
        }

class TimedPanguMLP(nn.Module):
    """Wrapper module inserting high-resolution stream synchronization points around original FFN layers."""
    def __init__(self, mlp, recorder: MLPTimingRecorder, layer_id: int):
        super().__init__()
        self.mlp = mlp
        self.recorder = recorder
        self.layer_id = layer_id

    def forward(self, hidden_states, *args, **kwargs):
        if (not self.recorder.enabled) or (not torch.npu.is_available()):
            return self.mlp(hidden_states, *args, **kwargs)

        L = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[0]
        phase = "decode" if L == 1 else "prefill"

        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)

        start_event.record()
        out = self.mlp(hidden_states, *args, **kwargs)
        end_event.record()

        self.recorder.add(start_event, end_event, self.layer_id, phase, 1)
        return out

def attach_mlp_timers(model, recorder: MLPTimingRecorder):
    """Hooks timing nodes across the network transformer blocks hierarchy."""
    for i, layer in enumerate(model.model.layers):
        layer.mlp = TimedPanguMLP(layer.mlp, recorder, layer_id=i)
    print(f"[INFO] Operational instrumentation hook deployed across {len(model.model.layers)} native layers.")


# ==============================================================================
# 5. Core Performance Measurement Engines
# ==============================================================================
@torch.no_grad()
def benchmark_peak_vram(model, prompt, tokenizer, max_new_tokens):
    """Measures maximum high-bandwidth memory usage during generation execution."""
    model.eval()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()

    with torch.amp.autocast("npu", dtype=torch.bfloat16):
        _ = model.generate(
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
    """Calculates steady-state generation latency and overall context throughput."""
    model.eval()
    
    # Warmup loop execution to ensure compilation state stability
    for _ in range(warmup_iters):
        with torch.amp.autocast("npu", dtype=torch.bfloat16):
            _ = model.generate(
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
    for _ in range(latency_iters):
        with torch.amp.autocast("npu", dtype=torch.bfloat16):
            _ = model.generate(
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

@torch.no_grad()
def benchmark_generate_with_mlp_profile(model, prompt, tokenizer, recorder: MLPTimingRecorder, max_new_tokens, profile_iters=5):
    """Executes structural layer profiling while preserving execution safety limits."""
    model.eval()
    recorder.reset()
    recorder.enabled = True
    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for _ in range(profile_iters):
        with torch.amp.autocast("npu", dtype=torch.bfloat16):
            _ = model.generate(
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
    recorder.enabled = False

    profiled_total_ms = start_event.elapsed_time(end_event)
    profiled_avg_latency_ms = profiled_total_ms / profile_iters

    num_layers = len(model.model.layers)
    mlp_summary = recorder.summarize(max_new_tokens, num_layers, profile_iters, profiled_avg_latency_ms)

    return profiled_avg_latency_ms, mlp_summary

@torch.no_grad()
def benchmark_manual_decode_only(model, prompt, tokenizer, max_new_tokens, profile_iters=5):
    """Isolates and evaluates the token generation (decode) loop by stripping prefill bias."""
    model.eval()

    input_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]

    with torch.amp.autocast("npu", dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    base_past_key_values = outputs.past_key_values
    first_next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)

    torch.npu.synchronize()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    total_decode_ms = 0.0

    for _ in range(profile_iters):
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

            with torch.amp.autocast("npu", dtype=torch.bfloat16):
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
# 6. Main Instrumentation Orchestrator Pipeline
# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    
    # --------------------------------------------------------------------------
    # Ascend Hardware Platform Environmental Setup (Milestone 1 Core Acceptance Logs)
    # --------------------------------------------------------------------------
    if torch.npu.is_available():
        device = torch.device("npu")
        
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
            
        try:
            torch.npu.set_per_process_memory_fraction(0.9)
        except Exception:
            pass
            
        # Dynamically retrieve hardware device name to pass compliance checks (Avoid static strings)
        device_name = torch.npu.get_device_name(device)
        print(f"[INFO] 硬件环境就绪：当前硬件为 【{device_name}】，本程序正基于昇腾运行。")
    else:
        device = torch.device("cpu")
        device_name = "CPU Backend"
        print(f"[WARNING] 硬件环境警告：未检测到昇腾计算硬件，回退至 【{device_name}】。测速指标将失效。")

    # Determine runtime data layout precision
    if args.dtype == "bf16":
        amp_dtype = torch.bfloat16
        load_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        amp_dtype = torch.float16
        load_dtype = torch.float16
    else:
        amp_dtype = torch.float32
        load_dtype = torch.float32

    train_args = CustomTrainingArgs()

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

    print("[INFO] Instantiating Dense Base Model weights layer topology...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        torch_dtype=load_dtype
    )
    model.resize_token_embeddings(len(tokenizer))
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[INFO] Restoring weights parameters from local checkpoint path: '{args.checkpoint}'") 
        from safetensors.torch import load_file
        state_dict = load_file(args.checkpoint, device="cpu")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    seed_all(42)

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
        
        # Phase 2: Complete Sequence Time-to-Last-Token End-to-End Analysis
        e2e_lat, e2e_thr = benchmark_e2e_generate_latency(
            model=model, prompt=prompt, tokenizer=tokenizer, 
            max_new_tokens=current_max_new_tokens, warmup_iters=10, latency_iters=50
        )
        e2e_latencies.append(e2e_lat)
        e2e_throughputs.append(e2e_thr)
        
        # Phase 3: Token Generation Inter-Token Isolated Decode Analysis
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
    print(f"dense_avg_peak_vram_mb = {avg_vram:.4f}")
    print(f"dense_avg_e2e_latency_ms = {avg_e2e_lat:.4f}")
    print(f"dense_avg_e2e_throughput = {avg_e2e_thr:.4f}")
    print(f"dense_avg_decode_latency_ms = {avg_dec_lat:.4f}")
    print(f"dense_avg_decode_throughput = {avg_dec_thr:.4f}")

    print("====================================================================================================\n")