# 🎙️ Auditory-TSE: 听觉注意力机制驱动的鸡尾酒会人声分离模型

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **模拟人类听觉注意力机制，解决"鸡尾酒会问题"**

在嘈杂的鸡尾酒会上，人类能够自然地聚焦于某个说话者，自动"过滤"掉背景噪音和其他人的对话。**Auditory-TSE** 使用深度学习模拟这一能力：给定目标说话者的一小段参考语音（注册音频），模型能够在多说话者混合音频中准确分离并提取出该说话者的声音。

---

## 🧠 核心架构

```
混合音频 [多人] ──→ 听觉前端(Conv1D+Gammatone) ──→ ┐
                                                    ├──→ 交叉注意力掩码估计 ──→ 解码器 ──→ 目标语音
注册音频 [目标人] ──→ 说话人编码器(ECAPA-TDNN) ──→ ┘         ↑ 自上而下注意
```

### 三项关键机制

1. **听觉前端**：可学习 Conv1D 编码器 + 可选 Gammatone 滤波器组（模拟耳蜗基底膜频率选择）
2. **说话人编码器**：ECAPA-TDNN 提取目标说话人的鲁棒声纹嵌入
3. **自上而下注意 (Top-Down Attention)**：通过 FiLM 条件调制和交叉注意力，用高层说话人身份信息调制低层声学特征的分离过程

### 可选的分离骨干网络

| 网络 | 特点 |
|---|---|
| **Conv-TasNet** | 高效时域卷积，生产级性能（默认） |
| **SepFormer** | 双路径 Transformer，更强长程建模 |
| **Cross-Attention** | 轻量交叉注意力掩码估计器 |

---

## 📦 快速开始

### 1. 安装

```bash
# 克隆项目（或将项目目录放入工作区）
cd 鸡尾酒会人声分离模型

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 快速验证

```bash
# 运行测试确保一切正常
pytest tests/ -v

# 快速查看模型结构
python -c "from src.models.auditory_tse import AuditoryTSE; m = AuditoryTSE(); print(f'{sum(p.numel() for p in m.parameters()):,} params')"
```

### 3. 合成数据演示

```bash
# 生成合成测试音频
python experiments/inference_demo.py --generate-samples

# 查看生成的音频
# demo_samples/demo_mixture.wav      — 双人混合
# demo_samples/demo_enrollment.wav   — 目标说话人参考
# demo_samples/demo_target_groundtruth.wav — 标注真值
```

---

## 🚀 训练

### 数据准备

```bash
# 下载 LibriMix 数据集
# 参照: https://github.com/JorisCos/LibriMix
# 将数据放入 data/LibriMix/
```

### 启动训练

```bash
# 默认配置（Conv-TasNet + LibriMix）
python experiments/train.py

# 使用 SepFormer
python experiments/train.py model=sepformer

# 自定义超参数
python experiments/train.py \
    training.learning_rate=5e-4 \
    training.epochs=200 \
    data.batch_size=16

# 多 GPU 训练（需要调整 trainer 代码中的 DataParallel/DDP）
python experiments/train.py data.batch_size=32
```

### 恢复训练

```bash
# 修改 trainer 代码中的 resume_from_checkpoint 调用，
# 或在代码中加载 checkpoint 后继续训练
```

---

## 📊 评估

```bash
# 在测试集上评估
python experiments/evaluate.py \
    checkpoint=checkpoints/auditory_tse_best_epoch0050.ckpt

# 保存分离音频
python experiments/evaluate.py \
    checkpoint=checkpoints/best.ckpt \
    save_audio=true
```

### 评估指标

| 指标 | 说明 | 理想值 |
|---|---|---|
| **SI-SDRi** (dB) | 尺度不变信噪比改善 | 越高越好 (> 10 dB) |
| **PESQ** | 感知语音质量 (ITU-T P.862) | 越高越好 (1.0 - 4.5) |
| **STOI** | 短时客观可懂度 | 越高越好 (0 - 1, > 0.75 良好) |

---

## 🎯 推理

```bash
# 分离指定说话者
python experiments/inference_demo.py \
    --input mixture.wav \
    --enrollment target_speaker.wav \
    --output separated.wav

# 使用指定模型
python experiments/inference_demo.py \
    --input mix.wav \
    --enrollment enroll.wav \
    --output out.wav \
    --checkpoint checkpoints/my_model.ckpt
```

### Python API

```python
from src.inference.pipeline import InferencePipeline

# 加载模型
pipeline = InferencePipeline.from_checkpoint("checkpoints/best.ckpt")

# 分离
separated = pipeline.run("mixture.wav", "enrollment.wav", "output.wav")

# 或者对 tensor 操作
import torchaudio
mix, sr = torchaudio.load("mixture.wav")
enr, _ = torchaudio.load("enrollment.wav")
result = pipeline.run_tensor(mix, enr)
```

---

## 📁 项目结构

```
├── configs/             # Hydra 层级配置
│   ├── model/           #   模型配置 (TasNet / SepFormer / AuditoryTSE)
│   ├── data/            #   数据配置 (LibriMix / WHAM!)
│   └── training/        #   训练配置 (默认 / 微调)
├── src/                 # 核心源码
│   ├── models/          #   模型：编码器、解码器、说话人编码器、分离网络
│   ├── audio/           #   音频处理：STFT、Gammatone、数据增强
│   ├── data/            #   数据加载：LibriMix / WHAM! 适配器
│   ├── training/        #   训练：循环、损失、优化器、回调
│   ├── evaluation/      #   评估：SI-SDRi, PESQ, STOI
│   └── inference/       #   推理管线
├── experiments/         # 可运行脚本
├── tests/               # 单元测试
└── notebooks/           # Jupyter 演示
```

---

## 🔧 开发

```bash
# 代码质量检查
ruff check src/ tests/
ruff format src/ tests/

# 运行测试
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html  # 覆盖率报告

# 只跑单元测试（跳过 GPU）
pytest tests/ -v -m "not gpu"
```

---

## 🔬 关键参考文献

- **Conv-TasNet**: Luo & Mesgarani (2019) — [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454)
- **SepFormer**: Subakan et al. (2021) — [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154)
- **ECAPA-TDNN**: Desplanques et al. (2020) — [ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification](https://arxiv.org/abs/2005.07143)
- **SpeakerBeam**: Delcroix et al. (2020) — [Improving Speaker Discrimination of Target Speech Extraction with Time-Domain SpeakerBeam](https://arxiv.org/abs/2001.08378)
- **TF-GridNet**: Wang et al. (2023) — [TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation](https://arxiv.org/abs/2209.03952)

---

## 📄 License

MIT License
