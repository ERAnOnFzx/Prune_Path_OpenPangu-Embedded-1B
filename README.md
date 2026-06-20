# PrunePath-OpenpanguMoE-1B

[English](README.md) | [简体中文](README_zh.md)

This repository contains the official open-source implementation for the paper **"PrunePath: Towards Highly Structured Sparse Language Models"**. Built on the Huawei Ascend NPU architecture, this project deeply integrates high-performance custom Triton operator kernels to achieve efficient, dynamic Sparse Mixture-of-Experts (MoE) injection and extreme inference acceleration for the openPangu large language model series.

---

## 📌 Key Features

* **Native Structural Flattening Optimization:** Directly intercepts and reconstructs the weight operators of traditional LLM MLP layers, removing high-overhead tensor cloning operations (`.clone()`) to significantly save memory bandwidth and space.
* **Physical Rearrangement Inference Acceleration:** Introduces an innovative, pre-inference physical space rearrangement algorithm for expert weights. This aligns perfectly with custom Triton core operators, achieving high-speed expert dynamic routing with zero CPU-NPU copy blockages.
* **Fully Fused SwiGLU Kernel:** Designed specifically for the Autoregressive Decode phase. This one-time fully fused SwiGLU kernel (completely fusing Gate -> Up -> Down) maximizes the parallel pipeline computing power of Ascend Tensor Cores.
* **Full Ascend Ecosystem Adaptation:** Deeply adapted to the MindSpore/Ascend ecosystem, with native support for `torch_npu` acceleration card drivers and `triton-ascend` operator compiler drivers.

---

## 🛠️ Environment Setup

To ensure absolute environment stability and avoid progress bar display issues when Conda calls Pip internally, please follow this strictly decoupled two-stage setup pipeline:

### Stage 1: Build the Base Virtual Environment
Create the foundational Conda environment containing underlying dynamic link libraries, the Python interpreter, and cross-platform core dependencies via `environment.yml`.

```bash
# Create the core environment image from the configuration file
conda env create -f environment.yml

# Activate the main runtime environment
conda activate pangu
```

### Stage 2: Complete the Computing Ecosystem and Third-Party Libraries
To monitor the downloading and compiling progress of advanced LLM ecosystem libraries in real-time, use Pip for an incremental installation after activating the environment.

```bash
# Install dependencies using Huawei's open-source mirror and dedicated heterogeneous source lock versions
pip install -r requirements.txt
```

---

## 📦 Model Checkpoints

We have published the fully sparse MoE weights of openPangu, after fine-tuning and rearrangement, on the [Hugging Face](https://huggingface.co/Zixun2408/PrunePath-OpenpanguMoE-1B/tree/main):

* **Base Dense Model:** `FreedomIntelligence/openPangu-Embedded-1B`
* **Downstream Fine-tuned Expert Models:** Please download `stage2_tau95_best.pt` or `stage2_tau70_best.pt` from our published repository and place them in the `./weights/` directory.

---

## 🚀 Execution Guide

### 1. Inference Pipeline (Text Summarization)

Use the following commands to initiate a single text summarization or batch document summarization processing flow.

**Mode A: Single-text Interactive Fast Inference**
```bash
python test_npu_xsum.py \
    --model_name "FreedomIntelligence/openPangu-Embedded-1B" \
    --checkpoint "./weights/stage2_tau95_best.pt" \
    --text "The input text documents go here..." \
    --tau 0.95
```

**Mode B: Large-batch Automated Offline Inference and Serialization**
```bash
python test_npu_xsum.py \
    --model_name "FreedomIntelligence/openPangu-Embedded-1B" \
    --checkpoint "./weights/stage2_tau95_best.pt" \
    --input_file "./data/val_docs.txt" \
    --output_file "./data/predictions.txt" \
    --tau 0.95
```

### 2. Evaluation Pipelines (Benchmarking & Performance)

We provide fully decoupled hardware-level profiling scripts (targeting pure Prefill/Decode latency, throughput, and HBM memory footprint) alongside a standard ROUGE metric validation flow.

| Track | Objective | Execution Command |
| :--- | :--- | :--- |
| **Track 1** | **Dense Baseline Analysis:** Measure the computation time distribution of the original model layers and native MLPs without sparse injection. | `python benchmark_dense.py --num_samples 10 --dtype bf16` |
| **Track 2** | **Native MoE Analysis:** Evaluate the dynamic routing expert overhead when using standard dynamic mask controls. | `python benchmark_origin.py --num_samples 10 --checkpoint "../stage2_tau70_best.pt"` |
| **Track 3** | **Triton Accelerated MoE:** Test extreme throughput performance after executing physical space rearrangement and mounting the highly fused Triton operators. | `python benchmark_triton.py --num_samples 10 --checkpoint "../weights/stage2_tau70_best.pt"` |
| **Track 4** | **XSum Task ROUGE Metric:** Quantitatively score the text semantic generation quality on the validation set, outputting standard ROUGE-1/2/L reports. | `python eval_xsum_rouge.py` |

---

## 📊 Hardware Compliance & Telemetry Statement

In accordance with compliance requirements for open-source and university collaboration projects, all automated scripts in this repository strictly reject hardcoded text logging during initialization and runtime. 

All hardware names, memory distributions, and target model identifiers in the runtime logs (`[INFO]` / `[SUCCESS]`) are dynamically bound from underlying native variables. The computing hardware relies entirely on `torch.npu.get_device_name(device)` for dynamic telemetry probing and explicitly outputs a "running on Ascend" safety declaration. The model loading system automatically binds the `args.model_name` variable to ensure accurate component audit information is logged. 

**Standard Compliance Log Example:**
```text
[INFO] 硬件环境检测：当前计算设备为 【NPU 0: Ascend910B1】，本程序正在基于昇腾运行。
[INFO] 模型组件装载：加载模型为openPangu 系列开源模型，当前目标模型变量名称/路径为: 'FreedomIntelligence/openPangu-Embedded-1B'
[SUCCESS] openPangu 系列模型组件与全量权重成功挂载至设备: 【NPU 0: Ascend910B1】
```

---

## 📂 Project Layout

```text
├── weights/                 # Fine-grained MoE weight expert checkpoints for various stages
├── environment.yml          # Base dependency and ARM-compatible environment build configuration
├── requirements.txt         # Ascend hardware driver (torch_npu) and LLM dependencies list
├── src/
├──── Pangu_Clusters/          # Auto-generated neuron Constrained K-Means clustering topology cache
├──── MoEBlock.py              # Core control layer for dynamic routing, Softmax gating, and numeric bound safety
├──── openpangu_moe.py         # Native structural flattening, dynamic MoE injection, and multi-task loss computation
├──── tl_kernel_optimized.py   # Hardware-level Triton kernel for Prefill prefix scanning and fully fused Decode SwiGLU
├──── test_npu_xsum.py         # Core entry script for single/batch offline accelerated summarization inference
├──── benchmark_dense.py       # Evaluation 1: Dense native Baseline architecture time profiling
├──── benchmark_origin.py      # Evaluation 2: Native MoE benchmarking with traditional dynamic mask control
├──── benchmark_triton.py      # Evaluation 3: Extreme acceleration benchmarking based on fully fused Triton kernels
└──── eval_xsum_rouge.py       # Evaluation 4: Standard ROUGE metric automated testing and quantitative reporting
```

---

## ✍️ Citation

If you use our codebase, algorithm design, or sparse weights in your academic research or engineering deliveries, please cite both our main work and the foundational 7B base model work:

```bibtex
@misc{gu2026prunepathhighlystructuredsparse,
      title={PrunePath: Towards Highly Structured Sparse Language Models}, 
      author={Zhexuan Gu and Zixun Fu and Yancheng Yuan},
      year={2026},
      eprint={2605.28283},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.28283}, 
}

@article{chen2025panguembedded,
      title={Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition},
      author={Chen, H and Wang, Y and Han, K and others},
      journal={arXiv preprint arXiv:2505.22375},
      year={2025}
}
```

---

## 📄 License

This project is open-sourced under the **Apache License 2.0**. For more detailed information, please refer to the `LICENSE` file within the repository.