# 先导入环境配置（必须第一行）
import env_setup
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --------------------------
# 配置参数
# --------------------------
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLING_RATE = 16000
# 测试音频路径（可以放一个自己的16kHz单声道WAV文件到data目录）
TEST_AUDIO_PATH = f"{env_setup.PROJECT_ROOT}/data/test.wav"

# --------------------------
# 加载模型和Processor（自动下载到data2的HF缓存）
# --------------------------
print("正在加载模型和Processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
print("✅ 模型加载完成！")

# --------------------------
# 音频预处理
# --------------------------
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

audio = load_audio(TEST_AUDIO_PATH, sr=SAMPLING_RATE)
inputs = processor(
    audio, 
    sampling_rate=SAMPLING_RATE, 
    return_tensors="pt", 
    padding=True
)

# --------------------------
# 推理
# --------------------------
print("正在进行语音识别...")
with torch.no_grad():
    logits = model(inputs.input_values).logits

pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]

print(f"\n识别结果: {transcription}")