# Wav2Vec2-Base Speech Recognition Finetuning Pipeline

本项目基于 Hugging Face Transformers 框架，实现了经典的 **Wav2Vec2** 自监督语音识别模型在 LibriSpeech（train-clean-100）数据集上的下游 CTC 微调。项目支持全离线闭环部署，在评估集上达到了 **3.76% 的超低字错率（WER）**。

---

## 🚀 项目核心亮点 & 异常攻坚

在搭建与训练流水线过程中，深入攻坚并解决了两项行业典型的语音工程与深度学习底层数值缺陷：

1. **音频解码层重构（破除 TorchCodec 依赖地狱）**
   - **问题缺陷**：在低版本 Linux 或环境污染的多版本 FFmpeg 下，Hugging Face 原生的 Arrow Audio 自动解码机制极易触发 `libtorchcodec` 底层 C++ 符号链接冲突（`undefined symbol: torch_from_blob`），导致程序死锁。
   - **解决方案**：摒弃了高层封装的 Dataset Audio 解码属性，通过内存级特征重构（`.to_dict()`），将数据集强行降级为纯 Python 基础字典，并引入 `soundfile` 针对音频物理路径执行硬解码。在保障**全系统零组件报错**的同时，将数据预处理映射速度拉升至 **64 examples/s**。

2. **混合精度（FP16）溢出治理（Loss 爆 NaN 参数冻结修复）**
   - **问题缺陷**：Wav2Vec2 在结合梯度检查点（Gradient Checkpointing）和混合精度（FP16）微调时，因 CTC 损失函数的动态数值范围超出 FP16 上限，极易发生数值溢出（Underflow），导致训练初期梯度范数暴盲（`grad_norm: NaN`）、训练损失（`loss: 0.0`），使模型权重参数遭遇实质性冻结。
   - **解决方案**：关闭 FP16 模式并动态切换为纯单精度（FP32）数据流或 BF16 稳健模式，同时将学习率从 `1e-4` 稳健下调至 `5e-5`。在重新清空损毁优化器状态后重启训练，彻底治愈了数值不稳定缺陷。

---

## 📦 项目目录结构

```text
wav2wev2base/
├── scripts/
│   └── finetune_wav2vec.py  # 核心微调与重构后的数据流水线脚本
├── .gitignore               # 大数据、缓存及模型权重隔离配置文件
└── README.md                # 项目全流程技术剖析与部署文档