import os
import sys
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

# ===================== 路径配置（和你项目对应） =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # /.../wav2wev2base/scripts
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)                 # /.../wav2wev2base
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "LibriSpeech")
SAVE_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "librispeech_clean")
SAMPLING_RATE = 16000

# 加入项目路径
sys.path.append(PROJECT_ROOT)

def read_librispeech_paths(data_path: Path):
    """只读取音频路径和文本，不提前加载数组，避免溢出"""
    all_samples = []
    # 遍历说话人文件夹
    for spk_dir in data_path.iterdir():
        if not spk_dir.is_dir():
            continue
        # 遍历章节文件夹
        for chap_dir in spk_dir.iterdir():
            if not chap_dir.is_dir():
                continue
            # 读取字幕文件
            txt_files = list(chap_dir.glob("*.txt"))
            if not txt_files:
                continue
            with open(txt_files[0], "r", encoding="utf-8") as f:
                trans_dict = {}
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    trans_dict[parts[0]] = parts[1]
            # 只保存音频路径，不加载数组
            for audio_file in chap_dir.glob("*.flac"):
                stem = audio_file.stem
                if stem not in trans_dict:
                    continue
                all_samples.append({
                    "audio_path": str(audio_file),
                    "text": trans_dict[stem]
                })
    return all_samples

if __name__ == "__main__":
    print("开始读取本地 LibriSpeech 数据...")

    # 读取训练集、测试集（只存路径，不加载音频）
    train_path = Path(DATA_ROOT) / "train-clean-100"
    test_path = Path(DATA_ROOT) / "test-clean"

    train_list = read_librispeech_paths(train_path)
    test_list = read_librispeech_paths(test_path)

    print(f"训练集样本数: {len(train_list)}")
    print(f"测试集样本数: {len(test_list)}")

    # 转为Dataset格式（此时audio_path是字符串，不会溢出）
    train_ds = Dataset.from_list(train_list)
    test_ds = Dataset.from_list(test_list)

    # 合并成一个dataset字典
    full_ds = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    # 关键步骤：让datasets自动加载音频并转为16kHz
    # 这里会把audio_path字段自动转为Audio类型，避免手动加载数组溢出
    full_ds = full_ds.cast_column("audio_path", Audio(sampling_rate=SAMPLING_RATE))
    # 重命名为audio字段，和原训练代码兼容
    full_ds = full_ds.rename_column("audio_path", "audio")

    # 持久化保存（后续训练直接加载，不用重复遍历文件）
    full_ds.save_to_disk(SAVE_DATA_DIR)

    print(f"\n✅ 本地数据处理完成！")
    print(f"数据集已保存至: {SAVE_DATA_DIR}")
    print("后续训练脚本直接用：load_from_disk 加载即可")