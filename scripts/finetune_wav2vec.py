# finetune_wav2vec.py
import os
import sys
import evaluate
import random
import jiwer
import torch
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, List, Union
from datasets import load_from_disk, DatasetDict, Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)

# ===================== 1. 全局路径 & 环境配置 =====================
# 🚀 核心重构：动态捕获当前脚本所在的项目根目录
# 假设 finetune_wav2vec.py 放在项目根目录下的 scripts/ 文件夹中
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # /.../wav2wev2base/scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                 # /.../wav2wev2base

# 自动拼接本地缓存目录，隐藏服务器私人路径
HF_CACHE = os.path.join(PROJECT_ROOT, "hf_cache")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE, "datasets")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.append(PROJECT_ROOT)
print(f"✅ 环境变量配置完成！当前动态感知项目根目录为: {PROJECT_ROOT}")
# ===================== 2. CTC 数据整理器 =====================
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [torch.tensor(f["input_values"]) for f in features]
        batch = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
        batch = {"input_values": batch}

        label_ids = [torch.tensor(f["labels"]) for f in features]
        labels = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
        batch["labels"] = labels
        return batch

# ===================== 3. 基础超参配置 =====================
MODEL_NAME = os.path.join(PROJECT_ROOT, "models", "wav2vec2-base-960h-offline")
SAMPLING_RATE = 16000
DATA_SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "librispeech_clean")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "wav2vec2-finetuned")

BATCH_SIZE = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3

# ===================== 4. 加载离线模型 & 处理器 =====================
print("\n[Step 4] 加载模型和Processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, local_files_only=True)
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    local_files_only=True
)
model.freeze_feature_encoder()
print("✅ 模型加载完成！")

# ===================== 5. 加载本地数据集 & 强力剥离特征（核心突破口） =====================
print("\n[Step 5] 加载本地预处理数据集...")
raw_dataset = load_from_disk(DATA_SAVE_DIR)

print("🔥 正在强制剥离 Arrow Audio 属性，彻底摧毁 torchcodec 触发路径...")
clean_dict = {}
for split in ["train", "test"]:
    # 1. 强制提取纯粹的 Python 基础类型数据（脱离 datasets schema 的管控）
    pure_audio_list = raw_dataset[split].to_dict()["audio"]
    pure_text_list = raw_dataset[split].to_dict()["text"]
    
    # 2. 重新构建成最干净、不带任何特殊 Feature 类型的纯数据 Dataset
    clean_dict[split] = Dataset.from_dict({
        "audio": pure_audio_list,
        "text": pure_text_list
    })

dataset = DatasetDict(clean_dict)
print("======== 纯进化数据结构检查 ========")
print(dataset)

# ===================== 6. 稳定版数据预处理函数 =====================
# ===================== 6. 稳定版数据预处理函数 =====================
# ===================== 6. 稳定版数据预处理函数 =====================
def preprocess_function(examples):
    audio_arrays = []
    
    # 🔥 核心修复：精准定位你的 LibriSpeech 物理根目录
    # 🔥 核心修复：将绝对路径替换为基于项目根目录的动态相对拼接
    # 无论在谁的电脑上，只要 data/LibriSpeech 结构不变，就能自适应精准定位
    AUDIO_REAL_ROOT = os.path.join(PROJECT_ROOT, "data", "LibriSpeech")

    for audio_item in examples["audio"]:
        # 兼容处理：获取纯文件名或相对路径
        if isinstance(audio_item, dict):
            audio_path = audio_item.get("path", "")
        else:
            audio_path = audio_item
            
        # 提取纯文件名，例如 "3374-298025-0025.flac"
        file_name = os.path.basename(audio_path)
        
        # 解析 LibriSpeech 的标准三级目录结构
        # 3374-298025-0025.flac -> ['3374', '298025', '0025.flac']
        parts = file_name.split("-")
        if len(parts) >= 2:
            reader_id = parts[0]    # 3374
            chapter_id = parts[1]   # 298025
            
            # 组合出完整的绝对物理路径
            # 优先尝试 train-clean-100 目录
            full_path = os.path.join(AUDIO_REAL_ROOT, "train-clean-100", reader_id, chapter_id, file_name)
            
            # 兜底保障：如果 train-clean-100 里没有，去 test-clean 目录里找（对应你的 test 集）
            if not os.path.exists(full_path):
                full_path = os.path.join(AUDIO_REAL_ROOT, "test-clean", reader_id, chapter_id, file_name)
        else:
            # 如果没有连字符，尝试直接拼接
            full_path = os.path.join(AUDIO_REAL_ROOT, audio_path)

        # 最终验证物理文件是否存在
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"\n❌ 依旧找不到真实的音频文件！"
                f"\n解析出的完整路径为: {full_path}"
                f"\n请确认该文件是否真实存在于服务器上。"
            )
            
        # 物理读取音频数据
        array, sampling_rate = sf.read(full_path)
        
        if len(array.shape) > 1:
            array = array[:, 0]  # 双声道转单声道
            
        audio_arrays.append(array)

    inputs = processor(
        audio_arrays,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
        padding=False
    )

    labels = processor.tokenizer(
        examples["text"],
        padding=False
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

print("\n[Step 6] 开始执行数据集预处理映射（此时不再有任何组件阻碍）...")
dataset = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=64,
    remove_columns=["audio", "text"]
)
print("✅ 数据集预处理映射完成！")

# ===================== 7. 初始化 DataCollator & 指标 =====================
data_collator = DataCollatorCTCWithPadding(processor=processor)
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # 将预测的 ID 序列解码为文本
    pred_str = processor.batch_decode(pred_ids)

    # 处理标签：将 padding 部分的 -100 替换回 pad_token_id 供解码
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 将真实标签 ID 序列解码为文本
    label_str = processor.batch_decode(
        label_ids,
        group_tokens=False
    )

    # 💡 核心修复：直接调用 jiwer 计算字错率 (Word Error Rate)
    wer = jiwer.wer(reference=label_str, hypothesis=pred_str)

    return {"wer": wer}
# ===================== 8. 训练参数配置 =====================
has_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != ""

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=has_gpu,  # 动态匹配，防 CPU 跑 fp16 崩溃
    gradient_checkpointing=True,
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    use_cpu=not has_gpu
)

# ===================== 9. 启动训练 =====================
print("\n[Step 9] 初始化 Trainer 并开始微调训练...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# 保存最终模型与处理器
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\n🎉 训练完成！微调模型已成功保存至: {OUTPUT_DIR}")