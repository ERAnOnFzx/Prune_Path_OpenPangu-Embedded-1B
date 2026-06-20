"""
OpenPangu-MoE XSum Summarization Inference Script (NPU Accelerated via Triton)
Compatible with Huawei openPangu University Cooperation Milestone 1 & 2 Acceptance Guidelines.

Reference Citation:
Chen H, Wang Y, Han K, et al. Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition.
arXiv preprint arXiv:2505.22375, 2025.
"""

import os
import torch
import torch_npu  # Must be imported immediately after torch
import argparse
from transformers import AutoConfig, AutoTokenizer
from openpangu_moe import CustomOpenPangu
from dataclasses import dataclass

# ==============================================================================
# 1. Global Constants & Trigger Tokens (Must align with training configuration)
# ==============================================================================
SUM_TOK = "<SUMMARY>"
DEFAULT_MAX_DOC_LENGTH = 400
DEFAULT_MAX_NEW_TOKENS = 64
TAU = 0.95

def parse_args():
    """Parses command line arguments for the inference task."""
    parser = argparse.ArgumentParser(description="OpenPangu-MoE XSum Summarization Inference on NPU")
    parser.add_argument("--model_name", type=str, default="FreedomIntelligence/openPangu-Embedded-1B", 
                        help="Name or local path of the pre-trained model")
    parser.add_argument("--checkpoint", type=str, default="../weights/stage2_tau95_best.pt", 
                        help="Path to the fine-tuned MoE weights")
    parser.add_argument("--text", type=str, default=(
                            "The ex-Reading defender denied fraudulent trading charges relating to the "
                            "Sodje Sports Foundation - a charity to raise money for Nigerian sport.\n"
                            "Mr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42.\n"
                            "Appearing at the Old Bailey earlier, all four denied the offence.\n"
                            "The charge relates to offences which allegedly took place between 2008 and 2014.\n"
                            "Sam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, "
                            "are due to stand trial in July.\nThey were all released on bail."
                        ), help="Direct text input for summarization")
    parser.add_argument("--input_file", type=str, default=None, 
                        help="Path to input file containing documents (one document per line)")
    parser.add_argument("--output_file", type=str, default=None, 
                        help="Path to save the generated summaries (defaults to console output)")
    parser.add_argument("--max_doc_length", type=int, default=DEFAULT_MAX_DOC_LENGTH, 
                        help="Maximum token length for input documents")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, 
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--tau", type=float, default=1.05,
                        help="Routing threshold parameter for MoE inference")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Initializes the environment, loads the model architecture, and restores fine-tuned weights."""
    
    # --------------------------------------------------------------------------
    # NPU Environment & Compliance Logging (Milestone 1 Requirements)
    # --------------------------------------------------------------------------
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    
    # Dynamic hardware verification via variables
    if torch.npu.is_available():
        device_name = torch.npu.get_device_name(device)
        print(f"[INFO] 硬件环境检测：当前计算设备为 【{device_name}】，本程序正在基于昇腾运行。")
    else:
        device_name = "CPU"
        print(f"[WARNING] 硬件环境警告：未检测到昇腾加速卡，当前运行于 【{device_name}】 环境。")

    # --------------------------------------------------------------------------
    # Tokenizer & Configuration Initialization
    # --------------------------------------------------------------------------
    print(f"[INFO] 模型初始化配置：加载模型为openPangu 系列开源模型，当前目标模型变量名称/路径为: '{args.model_name}'")
    
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Synchronize padding tokens
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Append special trigger tokens to prevent embedding dimensionality mismatch
    tokenizer.add_special_tokens({"additional_special_tokens": [SUM_TOK]})

    # --------------------------------------------------------------------------
    # Training Arguments Construction (Triton Kernel Optimization Enabled)
    # --------------------------------------------------------------------------
    @dataclass
    class custom_training_args:
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
        Triton: bool = True  # Enforce Triton kernel acceleration for dynamic routing

    train_args = custom_training_args()

    # --------------------------------------------------------------------------
    # Model Instantiation & Weight Restoration
    # --------------------------------------------------------------------------
    print(f"[INFO] 正在构建 CustomOpenPangu 架构 (Triton Acceleration = {train_args.Triton})...")
    model = CustomOpenPangu(
        model_name_or_path=args.model_name,
        config=config,
        train_args=train_args,
    )
    model.model.resize_token_embeddings(len(tokenizer))

    # Critical Step: Execute post-initialization prior to weight loading
    model.custom_post_init() 

    print(f"[INFO] 正在恢复细粒度微调权重，目标路径: '{args.checkpoint}'")
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        print(f"[WARNING] State_dict Missing keys (Top 5): {missing[:5]}")
    if unexpected:
        print(f"[WARNING] State_dict Unexpected keys (Top 5): {unexpected[:5]}")

    # Inference pre-computation: Physical routing reordering and precision deployment
    print(f"[INFO] 正在执行 MoE 专家网络推理前置物理重排...")
    model.physical_reorder_for_inference(args.tau)
    model.to(device, dtype=torch.bfloat16)
    model.eval()

    print(f"[SUCCESS] openPangu 系列模型组件与全量权重成功挂载至设备: 【{device_name}】")
    print(model)
    return model, tokenizer, device

def generate_summary(model, tokenizer, device, text, max_doc_length, max_new_tokens):
    """Executes beam-search text summarization for a single document input."""
    sum_tok_id = tokenizer.convert_tokens_to_ids(SUM_TOK)
    
    # Structure Context Prompt: [Document Tokens] + [<SUMMARY>]
    doc_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(doc_ids) > max_doc_length:
        doc_ids = doc_ids[:max_doc_length]
    prompt_ids = doc_ids + [sum_tok_id]

    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_attention_mask = torch.ones_like(prompt_input_ids)

    with torch.no_grad():
        generated_ids = model.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=4,  # Employ standard beam search to guarantee semantic generation quality
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = prompt_input_ids.shape[1]
    pred_ids = generated_ids[0, prompt_len:]
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
    return pred_text

def main():
    args = parse_args()
    
    # Parameter sanity check
    if not args.text and not args.input_file:
        raise ValueError("[ERROR] Missing inference source. Provide input via '--text' or '--input_file'.")

    # Initialize model pipeline
    model, tokenizer, device = load_model_and_tokenizer(args)
    outputs = []
    
    if args.text:
        # Standard Single-stream Inference Pipeline
        print("\n--- [Inference Task: Input Document Summary] ---")
        print(args.text)
        
        summary = generate_summary(model, tokenizer, device, args.text, args.max_doc_length, args.max_new_tokens)
        outputs.append(summary)
        
        print("\n--- [Inference Task: Generated Output Summary] ---")
        print(summary)
        print("-------------------------------------------------\n")
        
    elif args.input_file:
        # Batch File Processing Pipeline
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"[ERROR] Input file path does not exist: '{args.input_file}'")
            
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        print(f"\n[INFO] 批量推理启动：共读取到 {len(lines)} 条文档流，开始迭代生成...")
        for i, text in enumerate(lines):
            summary = generate_summary(model, tokenizer, device, text, args.max_doc_length, args.max_new_tokens)
            outputs.append(summary)
            print(f"[INFO] Progress: [{i+1}/{len(lines)}] -> Summary Generated Successfully.")

    # Serialize inference output records
    if args.output_file and outputs:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for out in outputs:
                f.write(out + "\n")
        print(f"[SUCCESS] 数据持久化完成：推理摘要已安全写入至 '{args.output_file}'")

if __name__ == "__main__":
    main()