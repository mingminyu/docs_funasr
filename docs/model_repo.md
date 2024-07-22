# 模型仓库

FunASR开源了大量在工业数据上预训练模型，您可以在模型许可协议下自由使用、复制、修改和分享 FunASR 模型，下面列举代表性的模型。

⭐ 表示ModelScope模型仓库，🤗 表示Huggingface模型仓库，🍀表示OpenAI模型仓库

## 1. 语音识别模型

|  模型名字      |        任务详情        |      训练数据      |  参数量   | 
|:------------------------------------:|:------------------:|:--------------:|:------:|
| SenseVoiceSmall <br> ([⭐](https://www.modelscope.cn/models/iic/SenseVoiceSmall)  [🤗](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) )   |  多种语音理解能力，涵盖了自动语音识别（ASR）、语言识别（LID）、情感识别（SER）以及音频事件检测（AED）   |  400000小时，中文   |  330M  |
| SeACoParaformer-zh <br> ( [⭐](https://www.modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) ) | 带热词功能的语音识别，带时间戳输出，非实时 |  60000小时，中文  | 220M |
| paraformer-zh <br> ([⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)  [🤗](https://huggingface.co/funasr/paraformer-zh) )    |  语音识别，带时间戳输出，非实时   |   60000小时，中文   |  220M  |
| paraformer-zh-spk <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary) )    |  分角色语音识别，带时间戳输出，非实时 |  60000小时，中文  | 220M |
| paraformer-zh-streaming <br> ( [⭐](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) [🤗](https://huggingface.co/funasr/paraformer-zh-streaming) ) |  语音识别，实时 |   60000小时，中文   |  220M  |
| paraformer-zh-streaming-small <br> ( [⭐](https://www.modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary)) |  语音识别，实时  |  60000小时，中文  | 220M |
|  paraformer-en <br> ( [⭐](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary) [🤗](https://huggingface.co/funasr/paraformer-en) )    |      语音识别，非实时      |   50000小时，英文   |  220M  |
| conformer-en <br> ( [⭐](https://modelscope.cn/models/damo/speech_conformer_asr-en-16k-vocab4199-pytorch/summary) [🤗](https://huggingface.co/funasr/conformer-en))  |  语音识别，非实时 | 50000小时，英文   |  220M  |
| ct-punc <br> ( [⭐](https://modelscope.cn/models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary) [🤗](https://huggingface.co/funasr/ct-punc) )   |  标点恢复  |   100M，中文与英文 |  290M  | 
| fsmn-vad <br> ( [⭐](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary) [🤗](https://huggingface.co/funasr/fsmn-vad))  |  语音端点检测，实时 | 5000小时，中文与英文 |  0.4M  | 
| fa-zh <br> ( [⭐](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary) [🤗](https://huggingface.co/funasr/fa-zh) )    |   字级别时间戳预测   |  50000小时，中文  |  38M   |
| cam++ <br> ( [⭐](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common/summary) [🤗](https://huggingface.co/funasr/campplus) )    |      说话人确认/分割      |     5000小时     |  7.2M  | 
| Whisper-large-v3 <br> ([⭐](https://www.modelscope.cn/models/iic/Whisper-large-v3/summary)  [🍀](https://github.com/openai/whisper) )   |  语音识别，带时间戳输出，非实时   |      多语言       | 1550 M |
| Qwen-Audio <br> ([⭐](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/qwen_audio/demo.py)  [🤗](https://huggingface.co/Qwen/Qwen-Audio) )  |  音频文本多模态大模型（预训练）   |      多语言       |   8B   |
| Qwen-Audio-Chat <br> ([⭐](https://github.com/modelscope/FunASR/blob/main/examples/industrial_data_pretraining/qwen_audio/demo_chat.py)  [🤗](https://huggingface.co/Qwen/Qwen-Audio-Chat) )   | 音频文本多模态大模型（chat版本） |   多语言  |   8B   |
| emotion2vec+large <br> ([⭐](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary)  [🤗](https://huggingface.co/emotion2vec/emotion2vec_plus_large) ) | 情感识别模型 | 40000小时，4种情感类别 |  300M  |
| [Paraformer-large](https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) |            非实时，输入wav文件持续时间不超过20秒             | 中文和英文，阿里巴巴语音数据（60000小时） | 8404，220M  |
| [Paraformer-large长音频版本](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary) |            非实时，能够处理任意长度的输入wav文件             | 中文和英文，阿里巴巴语音数据（60000小时） | 8404，220M  |
| [Paraformer-large-en长音频版本](https://www.modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020/summary) |            非实时，能够处理任意长度的输入wav文件             |    英文，阿里巴巴语音数据（50000小时）    | 10020，220M |
| [Paraformer-large-Spk](https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn/summary) |        非实时，在长音频功能的基础上添加说话人识别功能        | 中文和英文，阿里巴巴语音数据（60000小时） | 8404，220M  |
| [Paraformer-large热词](https://www.modelscope.cn/models/damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/summary) | 非实时，基于激励增强的热词定制支持，可以提高热词的召回率和准确率，输入wav文件持续时间不超过20秒 | 中文和英文，阿里巴巴语音数据（60000小时） | 8404，220M  |
| [Paraformer](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/summary) |             离线，输入wav文件持续时间不超过20秒              | 中文和英文，阿里巴巴语音数据（50000小时） |  8358，68M  |
| [Paraformer实时](https://modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online/summary) |                    实时，能够处理流式输入                    | 中文和英文，阿里巴巴语音数据 (50000hours) |  8404，68M  |
| [Paraformer-large实时](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary) |                    实时，能够处理流式输入                    | 中文和英文，阿里巴巴语音数据 (60000hours) | 8404，220M  |
| [Paraformer-tiny](https://www.modelscope.cn/models/damo/speech_paraformer-tiny-commandword_asr_nat-zh-cn-16k-vocab544-pytorch/summary) |      非实时，轻量级Paraformer模型，支持普通话命令词识别      |     中文，阿里巴巴语音数据 (200hours)     |  544，5.2M  |
| [Paraformer-aishell](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-aishell1-pytorch/summary) |                       非实时，学术模型                       |         中文，AISHELL (178hours)          |  4234，43M  |
| [ParaformerBert-aishell](https://modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell1-vocab4234-pytorch/summary) |                       非实时，学术模型                       |         中文，AISHELL (178hours)          |  4234，43M  |
| [Paraformer-aishell2](https://www.modelscope.cn/models/damo/speech_paraformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary) |                       非实时，学术模型                       |        中文，AISHELL-2 (1000hours)        |  5212，64M  |
| [ParaformerBert-aishell2](https://www.modelscope.cn/models/damo/speech_paraformerbert_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch/summary) |                       非实时，学术模型                       |        中文，AISHELL-2 (1000hours)        |  5212，64M  |

## 2. 多说话人语音识别模型

## 3. 语音端点检测模型

## 4. 标点恢复模型

## 5. 语音模型

## 6. 说话人确认模型

## 7. 说话人日志模型

|  模型名字        | 任务详情 |           训练数据            | 参数量 |
| :-----------: | :---: | :-----------: | :------: |
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-zh-cn-alimeeting-16k-n16k4-pytorch/summary) | | 中文，AliMeeting (120 小时) |  40.5M   |
| [SOND](https://www.modelscope.cn/models/damo/speech_diarization_sond-en-us-callhome-8k-n16k4-pytorch/summary) |  | 英文，CallHome (60 小时)   |   12M    |

## 8. 时间戳预测模型

|  模型名字        | 任务详情 |           训练数据            | 参数量 |
| :-----------: | :---: | :-----------: | :------: |
| [TP-Aligner](https://modelscope.cn/models/damo/speech_timestamp_prediction-v1-16k-offline/summary) | 时间戳模型  | 中文，阿里巴巴语音数据 (50000hours) |  37.8M   |

## 9. 逆文本正则化

