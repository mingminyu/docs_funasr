# 功能概览


## 2. 语音端点检测

### 2.1 非实时



### 2.2 实时

```python linenums="1"
import soundfile
from funasr import AutoModel

chunk_size = 200 # ms
model = AutoModel(model="fsmn-vad")


wav_file = f"{model.model_path}/example/vad_example.wav"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = int(chunk_size * sample_rate / 1000)

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size)
    if len(res[0]["value"]):
        print(res)
```

流式 VAD 模型输出格式为4种情况：

- [[beg1, end1], [beg2, end2], .., [begN, endN]]：同上离线VAD输出结果。
- [[beg, -1]]：表示只检测到起始点。
- [[-1, end]]：表示只检测到结束点。
- []：表示既没有检测到起始点，也没有检测到结束点 输出结果单位为毫秒，从起始点开始的绝对时间。


## 3. 标点恢复

```python linenums="1"
from funasr import AutoModel

model = AutoModel(model="ct-punc")

res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
print(res)
```

## 4. 情感识别

```python linenums="1"
from funasr import AutoModel

model = AutoModel(model="fa-zh")
wav_file = f"{model.model_path}/example/asr_example.wav"
text_file = f"{model.model_path}/example/text.txt"
res = model.generate(input=(wav_file, text_file), data_type=("sound", "text"))
print(res)
```

## 5. 情感识别

```python linenums="1"
from funasr import AutoModel

model = AutoModel(model="emotion2vec_plus_large")
wav_file = f"{model.model_path}/example/test.wav"
res = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
print(res)
```

