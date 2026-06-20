"""
OpenPangu-MoE XSum Summarization Generation & ROUGE Evaluation Pipeline
Implements Automated Benchmarking, Metrics Analysis, and Compliance Logging on Ascend NPU.

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
import torch
import torch_npu  # Strict execution order: Must be imported immediately following torch
import evaluate
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset
from openpangu_moe import CustomOpenPangu
from dataclasses import dataclass

# ==============================================================================
# 1. Global Performance Testing Constants & Evaluation Hyperparameters
# ==============================================================================
MAX_DOC_LENGTH = 400
SUM_TOK = "<SUMMARY>"
MODEL_NAME = "FreedomIntelligence/openPangu-Embedded-1B"
WEIGHT_PATH = "../weights/stage2_tau95_best.pt"
TAU = 0.95
EVAL_SAMPLES = 100  # Target batch partition limit for dataset profiling

# ==============================================================================
# 2. Ascend Hardware Platform Environmental Setup (Milestone 1 Core Acceptance Logs)
# ==============================================================================
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
device = torch.device("npu:0" if torch.npu.is_available() else "cpu")

if torch.npu.is_available():
    device_name = torch.npu.get_device_name(device)
    print(f"[INFO] 硬件环境检测：当前计算设备为 【{device_name}】，本程序正在基于昇腾运行。")
else:
    device_name = "CPU Backend"
    print(f"[WARNING] 硬件环境警告：未检测到昇腾加速卡，回退至 【{device_name}】 环境。")

# ==============================================================================
# 3. Component Tokenizer & Dataset Initialization Pipeline
# ==============================================================================
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Synchronize padding sequences configuration
tokenizer.pad_token = tokenizer.eos_token
config.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Critical Path: Append training-phase trigger keywords into active tokenizer space
tokenizer.add_special_tokens({"additional_special_tokens": [SUM_TOK]})

print("[INFO] Fetching evaluation dataset split (XSum: Validation Partition)...")
raw_dataset = load_dataset("xsum")
valid_dataset = raw_dataset["validation"]

# Slice and bind the targeted profiling subset bounds
limit_size = min(EVAL_SAMPLES, len(valid_dataset))
eval_dataset = valid_dataset.select(range(limit_size))
print(f"[INFO] Evaluation partition locked. Processing target cohort of {limit_size} sample streams.")

# ==============================================================================
# 4. Hyperparameter Argument Alignment Layer for MoE Blocks
# ==============================================================================
@dataclass
class CustomTrainingArgs:
    """Standardized configuration state layer for MoE infrastructure synchronization."""
    stage1_epoch: int = 1
    stage2_epoch: int = 1
    batch: int = 1
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
    tau: float = TAU
    stage1_tau: float = 1.05
    stage2_tau: float = 0.95

train_args = CustomTrainingArgs()

# ==============================================================================
# 5. Core Model Architecture Loading & Fine-Tuned Weights Restoration
# ==============================================================================
print(f"[INFO] 模型组件装载：加载模型为openPangu 系列开源模型，当前目标模型变量名称/路径为: '{MODEL_NAME}'")
model = CustomOpenPangu(
    model_name_or_path=MODEL_NAME,
    config=config,
    train_args=train_args,
)
model.model.resize_token_embeddings(len(tokenizer))

# Critical Execution Topology Order: Process post-initializations prior to weight injection
model.custom_post_init() 

print(f"[INFO] 正在恢复细粒度微调权重，目标路径: '{WEIGHT_PATH}'")
state_dict = torch.load(WEIGHT_PATH, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)

if missing:
    print(f"[WARNING] State_dict Missing keys (Top 5): {missing[:5]}")
if unexpected:
    print(f"[WARNING] State_dict Unexpected keys (Top 5): {unexpected[:5]}")

# Inference Pre-computation: Physically reorder routing metrics and deploy to NPU accelerator
print("[INFO] 正在执行 MoE 专家网络推理前置物理重排...")
model.physical_reorder_for_inference(TAU)
model.to(device, dtype=torch.bfloat16)
model.eval()
print(f"[SUCCESS] openPangu 系列模型组件与全量权重成功挂载至设备: 【{device_name}】")

# ==============================================================================
# 6. Quantitative Generation Loop & Downstream Reference Evaluation
# ==============================================================================
sum_tok_id = tokenizer.convert_tokens_to_ids(SUM_TOK)
rouge = evaluate.load("rouge")

predictions = []
references = []

print("\n[INFO] Pipeline execution initialized: Auto-regressive Generation & ROUGE Evaluation starting.")
profiler_progress_bar = tqdm(eval_dataset, desc="Evaluating openPangu-MoE (XSum)")

for i, sample in enumerate(profiler_progress_bar):
    doc = sample["document"]
    ref_summary = sample["summary"]
    references.append(ref_summary)

    # Format structured context prompt: [Document Sequence] + [<SUMMARY>]
    doc_ids = tokenizer.encode(doc, add_special_tokens=False)
    if len(doc_ids) > MAX_DOC_LENGTH:
        doc_ids = doc_ids[:MAX_DOC_LENGTH]
    prompt_ids = doc_ids + [sum_tok_id]

    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_attention_mask = torch.ones_like(prompt_input_ids)

    with torch.no_grad():
        # Execute generating sequence using standardized hyperparameter beam search configurations
        generated_ids = model.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=64,
            num_beams=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = prompt_input_ids.shape[1]
    pred_ids = generated_ids[0, prompt_len:]
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
    predictions.append(pred_text)

    # Telemetry logging: Output verification metrics tracking across the initial 3 operational iterations
    if i < 3:
        print(f"\n=================== [Operational Log: Step {i + 1} Pipeline Telemetry] ===================")
        print(f"[*] Stream Index Target : Sample {i + 1}")
        print(f"[*] Target Ground Truth : {ref_summary}")
        print(f"[*] Model Generation    : {pred_text}")
        print("=======================================================================================\n")

# ==============================================================================
# 7. Post-Processing Statistics Serialization & Final Report Extraction
# ==============================================================================
print("[INFO] Generation cycle completed. Computing cross-entropy ROUGE statistics mappings...")
results = rouge.compute(predictions=predictions, references=references)

print("\n======================= 📈 ROUGE 评测最终报告 =======================")
print(f"### Target Model Identifier : {MODEL_NAME}")
print(f"### Acceleration Hardware   : {device_name}")
print(f"### Cohort Sample Evaluated : {limit_size} operational streams")
print("---------------------------------------------------------------------")
print(f"### ROUGE-1 Metric Score    : {results['rouge1'] * 100:.2f}")
print(f"### ROUGE-2 Metric Score    : {results['rouge2'] * 100:.2f}")
print(f"### ROUGE-L Metric Score    : {results['rougeL'] * 100:.2f}")
print(f"### ROUGE-Lsum Metric Score : {results['rougeLsum'] * 100:.2f}")
print("=====================================================================\n")