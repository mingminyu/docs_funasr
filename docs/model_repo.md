# 模型仓库

FunASR开源了大量在工业数据上预训练模型，您可以在[模型许可协议](./MODEL_LICENSE)下自由使用、复制、修改和分享FunASR模型，下面列举代表性的模型，更多模型请参考 [模型仓库](./model_zoo)。

⭐ 表示ModelScope模型仓库，🤗 表示Huggingface模型仓库，🍀表示OpenAI模型仓库


|                                                                                                     模型名字                                                                                                      |        任务详情        |      训练数据      |  参数量   | 
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------:|:--------------:|:------:|
|   SenseVoiceSmall <br> ([⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall)  [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) )   |  多种语音理解能力，涵盖了自动语音识别（ASR）、语言识别（LID）、情感识别（SER）以及音频事件检测（AED）   |  400000小时，中文   |  330M  |
|    paraformer-zh <br> ([⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)  [🤗](https://huggingface.co/funasr/paraformer-zh) )    |  语音识别，带时间戳输出，非实时   |   60000小时，中文   |  220M  |
| paraformer-zh-streaming <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) ) |      语音识别，实时       |   60000小时，中文   |  220M  |
|         paraformer-en <br> ( [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary) [🤗](https://huggingface.co/funasr/paraformer-en) )         |      语音识别，非实时      |   50000小时，英文   |  220M  |
|                      conformer-en <br> ( [⭐](https://modelscope.cn/models/damo/speech_conformer_asr-en-16k-vocab4199-pytorch/summary) [🤗](https://huggingface.co/funasr/conformer-en) )                      |      语音识别，非实时      |   50000小时，英文   |  220M  |
|                        ct-punc <br> ( [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) )                         |        标点恢复        |   100M，中文与英文   |  290M  | 
|                            fsmn-vad <br> ( [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad) )                             |     语音端点检测，实时      |  5000小时，中文与英文  |  0.4M  | 
|                              fa-zh <br> ( [⭐](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary) [🤗](https://huggingface.co/funasr/fa-zh) )                               |      字级别时间戳预测      |   50000小时，中文   |  38M   |
|                                 cam++ <br> ( [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) )                                 |      说话人确认/分割      |     5000小时     |  7.2M  | 
|                                     Whisper-large-v3 <br> ([⭐](https://www.modelscope.cn/models/iic/Whisper-large-v3/summary)  [🍀](https://github.com/openai/whisper) )                                      |  语音识别，带时间戳输出，非实时   |      多语言       | 1550 M |
|                                         Qwen-Audio <br> ([⭐](examples/industrial_data_pretraining/qwen_audio/demo.py)  [🤗](https://huggingface.co/Qwen/Qwen-Audio) )                                         |  音频文本多模态大模型（预训练）   |      多语言       |   8B   |
|                                 Qwen-Audio-Chat <br> ([⭐](examples/industrial_data_pretraining/qwen_audio/demo_chat.py)  [🤗](https://huggingface.co/Qwen/Qwen-Audio-Chat) )                                  | 音频文本多模态大模型（chat版本） |      多语言       |   8B   |
|                        emotion2vec+large <br> ([⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary)  [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) )                        |    情感识别模型          | 40000小时，4种情感类别 |  300M  |
