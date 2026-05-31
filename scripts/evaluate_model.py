import env_setup
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# --------------------------
# 配置参数
# --------------------------
# 微调后的模型路径
FINETUNED_MODEL_PATH = f"{env_setup.PROJECT_ROOT}/models/wav2vec2-finetuned"
SAMPLING_RATE = 16000
# 测试音频路径（可以用多个音频测试）
TEST_AUDIO_PATH = f"{env_setup.PROJECT_ROOT}/data/test.wav"

# --------------------------
# 加载微调后的模型
# --------------------------
print("加载微调后的模型...")
processor = Wav2Vec2Processor.from_pretrained(FINETUNED_MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(FINETUNED_MODEL_PATH)
print("✅ 模型加载完成！")

# --------------------------
# 测试识别效果
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

print("正在识别...")
with torch.no_grad():
    logits = model(inputs.input_values).logits

pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]

print(f"\n微调后模型识别结果: {transcription}")
print("（可对比微调前的结果，说明微调效果）")