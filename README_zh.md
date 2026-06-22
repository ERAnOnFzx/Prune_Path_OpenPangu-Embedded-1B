# PrunePath-OpenpanguMoE-1B

[English](README.md) | [简体中文](README_zh.md)

本仓库是论文 **"PrunePath: Towards Highly Structured Sparse Language Models"** 的官方开源代码实现。本项目基于华为昇腾计算硬件（Ascend NPU）底座，深度集成了 Triton 高性能自定义算子内核，实现了针对大模型 openPangu 系列开源模型的高效、动态稀疏混合专家架构（MoE）注入与推理加速。

---

## 📌 项目特性

* **原生结构打平优化：** 直接接管并重构传统大模型 MLP 层的权重算子，移除高开销的张量克隆（`.clone()`），大幅节省显存带宽与空间。
* **物理重排推理加速：** 首创推理前置的专家权重物理空间重排算法，完美对齐自定义 Triton 核心算子，实现零 CPU-NPU 拷贝阻塞的高速专家动态路由。
* **全融合 SwiGLU 内核：** 针对 Autoregressive Decode 阶段，设计了一次性全融合的 SwiGLU 内核（Gate -> Up -> Down 彻底融合），全面压榨昇腾 Tensor Core 的并行流水线算力。
* **全量国产化生态适配：** 深度适配昇腾昇思（MindSpore）生态，原生支持 `torch_npu` 加速卡驱动与 `triton-ascend` 算子编译器驱动。

---

## 🛠️ 环境协同配置

为了保证环境装载的绝对稳定，并规避由于 Conda 内部调用 Pip 时不显示进度条的问题，请严格按照以下双阶段解耦流水线进行环境配置：

### 阶段一：基础虚拟环境构建
首先通过 `environment.yml` 构建包含底层动态链接库、Python 解释器及跨平台核心依赖的基础 Conda 环境。

```
# 从配置文件创建核心环境镜像
conda env create -f environment.yml

# 激活主运行环境
conda activate pangu
```

### 阶段二：计算生态与第三方库补全
为了实时监控高阶大语言模型生态库的下载与编译进度，激活环境后，使用 Pip 执行增量安装。

``` bash
# 利用华为开源镜像及专用异构源锁版本安装第三方依赖
pip install -r requirements.txt
```

---

## 📦 模型权重获取

我们已将微调及重排后的 openPangu-MoE 全量稀疏权重发布至 [Hugging Face](https://huggingface.co/Zixun2408/PrunePath-OpenpanguMoE-1B/tree/main) 模型社区：

* **基础稠密底座模型：** `FreedomIntelligence/openPangu-Embedded-1B`
* **下游微调专家模型：** 请从我们发布的存储库下载权重（默认设置中使用的是 `stage2_tau95_best.pt` 或 `stage2_tau70_best.pt`） 并置于 `./weights/` 目录下。

---

## 🚀 运行指南

### 1. 文本摘要生成推理

使用以下命令启动单条文本摘要生成。

**单条文本交互式推理**
```
python inference.py \
    --model_name "FreedomIntelligence/openPangu-Embedded-1B" \
    --checkpoint "./weights/stage2_tau95_best.pt" \
    --text "The input text documents go here..." \
    --tau 0.95
```

### 2. 基准测速与多维度性能评估

我们提供了三个完全解耦的硬件级测速与剖析脚本（针对纯净 Prefill/Decode 延迟、吞吐量和 HBM 显存占用），以及标准的 ROUGE 指标校验流。

| 评测轨道 | 目标说明 | 执行命令 |
| :--- | :--- | :--- |
| **轨道一** | **稠密 Baseline 性能分析：** 测量未经稀疏化注入的原始模型层及原生 MLP 的计算耗时分布。 | `python benchmark_dense.py --num_samples 10 --dtype bf16` |
| **轨道二** | **传统原生 MoE 性能分析：** 评估使用常规动态掩码控制时的动态路由专家开销。 | `python benchmark_origin.py --num_samples 10 --checkpoint "../stage2_tau70_best.pt" --tau 0.70` |
| **轨道三** | **Triton 内核全加速 MoE 性能分析：** 测试执行物理空间重排、并完全挂载 Triton 高融合算子后的极致吞吐性能。 | `python benchmark_triton.py --num_samples 10 --checkpoint "../weights/stage2_tau70_best.pt" --tau 0.70` |
| **轨道四** | **XSum 下游摘要任务 ROUGE 评测：** 对模型在验证集上的文本语义生成质量进行定量打分，自动输出 ROUGE-1/2/L 指标最终报告。 | `python test_npu_rougel.py --checkpoint "../stage2_tau70_best.pt" --tau 0.70` |

---

## 📊 硬件兼容性与合规性声明

根据开源及高校合作项目的合规性要求，本仓库所有自动化脚本在初始化和运行阶段拒绝任何硬编码固定文本打印。

所有运行时日志（`[INFO]` / `[SUCCESS]`）中的硬件卡名称、显存分布和目标模型标识符均绑定自底层的原生变量。计算硬件完全依赖 `torch.npu.get_device_name(device)` 进行动态 Telemetry 探测，并显式输出其“基于昇腾运行”的安全声明。模型载入体系自动绑定 `args.model_name` 变量，如实将包含“加载模型为openPangu 系列开源模型”的组件审计信息载入系统日志。

**标准的合规运行日志样例如下：**
```
[INFO] 硬件环境检测：当前计算设备为 【NPU 0: Ascend910B1】，本程序正在基于昇腾运行。
[INFO] 模型组件装载：加载模型为openPangu 系列开源模型，当前目标模型变量名称/路径为: 'FreedomIntelligence/openPangu-Embedded-1B'
[SUCCESS] openPangu 系列模型组件与全量权重成功挂载至设备: 【NPU 0: Ascend910B1】
```

---

## 📂 仓库架构说明

```text
├── weights/                 # 细粒度微调后的各阶段 MoE 权重专家检查点文件
├── environment.yml          # 国产化基础依赖与 ARM 兼容环境编译镜像配置文件
├── requirements.txt         # 昇腾特殊硬件驱动（torch_npu）及大模型依赖补全列表
├── src/          
├──── Pangu_Clusters/                         # 自动化生成的神经元 Constrained K-Means 聚类拓扑缓存
├──── MoEBlock.py                             # 动态路由、Softmax 门控与数值边界安全防护核心控制层
├──── openpangu_moe.py                        # 原生大模型结构打平、MoE 动态注入与多任务损失综合计算模块
├──── MoEBlock_origin.py                      # 传统动态掩码控制的动态路由、Softmax 门控与数值边界安全防护核心控制层
├──── openpangu_moe_origin.py                 # 传统动态掩码控制的原生大模型结构打平、MoE 动态注入与多任务损失综合计算模块
├──── tl_kernel_optimized_Ultimate_PanGU.py   # Triton 硬件级 Prefill 前缀扫描与 Decode 全融合 SwiGLU 自定义算子内核
├──── inference.py                            # 单条摘要生成脚本
├──── evaluation_dense.py                      # 评测代码 1：Dense 原生 Baseline 架构耗时细分剖析脚本
├──── evaluation_origin.py                     # 评测代码 2：传统动态掩码控制的 MoE 基础测速脚本
├──── evaluation_triton.py                     # 评测代码 3：基于 Triton 全融合内核的极致加速测速脚本
└──── test_npu_rougel.py                        # 评测代码 4：标准 ROUGE 指标自动化测试与定量报告流
```

---

## ✍️ 论文引用

如果您在学术研究、工程交付中使用了我们的代码库、算法设计或稀疏权重，请务必同时引用我们的主工作及基础 7B 基座工作：

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

## 📄 开源许可证

本项目基于 **Apache License 2.0** 许可证开源。有关详细信息，请参阅项目内部的 `LICENSE` 文件。
