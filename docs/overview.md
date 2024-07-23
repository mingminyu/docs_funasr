# 功能概览

## 1. 语音识别

### 1.1 非实时语音识别

下面我们使用 SenseVoice 和 Paraformer 模型对音频进行识别:

???+ example "非实时语音识别"

    === "SenseVoice"

        ```python linenums="1"
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        model_dir = "iic/SenseVoiceSmall"

        model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )

        # en
        res = model.generate(
            input=f"{model.model_path}/example/en.mp3",
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        print(text)
        ```

        参数说明:
        
        - `model_dir`: 模型名称或本地模型路径
        - `vad_model`: 开启 VAD。
        - `vad_kwargs`: VAD 模型配置，`max_single_segment_time` 表示最大切割音频时长, 单位是毫秒。
        - `use_itn`: 输出结果中是否包含标点与逆文本正则化。
        - `batch_size_s`: 启用动态 batch，每个 batch中总音频时长，单位为秒。
        - `merge_vad`: 是否将 vad 模型切割的短音频碎片合成，合并后长度为`merge_length_s`，单位为秒。
        - `ban_emo_unk`: 禁用 `emo_unk` 标签，禁用后所有的句子都会被赋与情感标签。

    === "Paraformer"

        ```python linenums="1"
        from funasr import AutoModel
        # paraformer-zh is a multi-functional asr model
        # use vad, punc, spk or not as you need
        model = AutoModel(
          model="paraformer-zh", 
          vad_model="fsmn-vad",
          punc_model="ct-punc", 
          # spk_model="cam++"
              )
        res = model.generate(
          input=f"{model.model_path}/example/asr_example.wav", 
          batch_size_s=300, 
          hotword='魔搭'
          )
        print(res)
        ```

### 1.2 实时语音识别

```python linenums="1"
import os
import soundfile
from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="paraformer-zh-streaming")

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960 # 600ms

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    print(res)
```

!!! note "代码说明"

    - `chunk_size` 为流式延时配置，`[0,10,5]` 表示上屏实时出字粒度为 10x60=600ms，未来信息为 5x60=300ms。每次推理输入为 600ms（采样点数为16000x0.6=960），输出为对应文字，最后一个语音片段输入需要设置 `is_final=True` 来强制输出最后一个字。

## 2. 语音端点检测

### 2.1 非实时

```python linenums="1"
from funasr import AutoModel

model = AutoModel(model="fsmn-vad")
wav_file = f"{model.model_path}/example/vad_example.wav"
res = model.generate(input=wav_file)
print(res)
```

VAD 模型输出格式为：[[beg1, end1], [beg2, end2], .., [begN, endN]]，其中 begN/endN 表示第 N 个有效音频片段的起始点/结束点， 单位为毫秒。

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

