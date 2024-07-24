# 语音端点检测

语音端点检测通常称为 VAD，用于检测有声音频片段。FunASR 中通过 fsmn-vad 模型进行调用，支持实时和非实时。

## 1. 非实时 VAD

非实时的 VAD 检测在 Python 中调用很简单，其输出格式是一个列表，每个元素为一个有效音频片段，包含开始和结束时间，单位为毫秒。

```python linenums="1"
from funasr import AutoModel

model = AutoModel(model="fsmn-vad")
wav_file = f"{model.model_path}/example/vad_example.wav"
res = model.generate(input=wav_file)
print(res)
```

## 2. 实时 VAD




