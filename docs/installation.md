# 安装

## 1. PIP

建议先创建一个虚拟环境，避免干扰系统环境

```bash
pip install torch torchaudio funasr
```

如果需要使用工业预训练模型，还需要安装 modelscope 与 huggingface_hub。

```bash
pip install -U modelscope huggingface huggingface_hub
```

此外，因为 FunASR 依赖 ffmpeg 对音频进行处理，如果你使用的 conda 环境，我们建议您在 conda 环境中安装 ffmpeg:

```bash
conda install ffmpeg
```

## 2. 源码

通过以下命令 FunASR 源码安装:

```bash
git clone https://github.com/alibaba/FunASR.git && cd FunASR
pip3 install -e ./
```

## 3. Docker

