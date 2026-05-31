import os
import sys

# --------------------------
# 1. 强制设置所有缓存/镜像路径（一劳永逸）
# --------------------------
# Hugging Face 模型/数据集缓存 → 存到data2
os.environ["HF_HOME"] = "/home/gzhu/data2/zx/wav2wev2base"
os.environ["HF_DATASETS_CACHE"] = "/home/gzhu/data2/zx/wav2wev2base/datasets"
# 国内镜像加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_CACHE"] = "/home/gzhu/data2/zx/wav2wev2base/transformers"
# --------------------------
# 2. 项目根目录路径（方便后续文件引用）
# --------------------------
PROJECT_ROOT = "/home/gzhu/data2/zx/wav2wev2base"
sys.path.append(PROJECT_ROOT)  # 把项目根目录加入Python路径

print("✅ 环境变量配置完成！")
print(f"项目根目录: {PROJECT_ROOT}")
print(f"HF缓存目录: {os.environ['HF_HOME']}")