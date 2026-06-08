# Project: 鸡尾酒会人声分离模型

> **Auditory-TSE**: Auditory-inspired Target Speaker Extraction — 模拟人类听觉注意力机制的多说话者语音分离系统。

---

## Identity

本项目构建一个基于深度学习的语音分离系统，核心目标是解决"鸡尾酒会问题"（Cocktail Party Problem）：在多人同时交谈的环境中，准确分离并跟踪特定目标说话者的声音。

**核心创新点**：模拟人类听觉系统中的**自上而下注意力机制（Top-Down Auditory Attention）**——利用目标说话人的声纹特征作为高层引导信号，调制低层声学特征的分离过程。

---

## Tech Stack

| 类别 | 技术 | 版本要求 |
|---|---|---|
| 语言 | Python | ≥ 3.10 |
| 深度学习 | PyTorch + torchaudio | ≥ 2.5.0 |
| 音频分离 | Asteroid | ≥ 0.7.0 |
| 音频分析 | librosa, soundfile | latest |
| 配置管理 | Hydra + OmegaConf | ≥ 1.3 |
| 实验追踪 | TensorBoard | ≥ 2.14 |
| 代码质量 | ruff | ≥ 0.3 |
| 测试 | pytest | ≥ 8.0 |
| 环境 | Windows 11 / Linux | — |

---

## Project Structure

```
鸡尾酒会人声分离模型/
├── CLAUDE.md                         # This file — project constitution
├── README.md                         # Project documentation
├── pyproject.toml                    # Project metadata + tool configs
├── requirements.txt                  # Python dependencies
│
├── configs/                          # Hydra hierarchical configs
│   ├── config.yaml                   # Main entry point
│   ├── model/                        # Model-specific configs
│   │   ├── auditory_tse.yaml         #   Auditory-TSE (default)
│   │   ├── tasnet.yaml               #   Conv-TasNet baseline
│   │   └── sepformer.yaml            #   SepFormer (transformer)
│   ├── data/                         # Dataset configs
│   │   ├── librimix.yaml
│   │   └── wham.yaml
│   ├── training/                     # Training configs
│   │   ├── default.yaml
│   │   └── finetune.yaml
│   └── inference/                    # Inference configs
│       └── default.yaml
│
├── src/                              # Core source code
│   ├── __init__.py
│   ├── models/                       # Neural network models
│   │   ├── encoder.py                #   Auditory encoder (Conv1D + Gammatone)
│   │   ├── decoder.py                #   Audio decoder (Transposed Conv)
│   │   ├── speaker_encoder.py        #   Speaker embedding (ECAPA-TDNN)
│   │   ├── separation/               #   Separation networks
│   │   │   ├── conv_tasnet.py        #     Conv-TasNet
│   │   │   ├── sepformer.py          #     SepFormer (dual-path transformer)
│   │   │   └── cross_attn_mask.py    #     Cross-attention mask estimator
│   │   └── auditory_tse.py           #   Main Auditory-TSE model
│   ├── audio/                        # Audio processing utilities
│   │   ├── transforms.py             #   STFT/iSTFT, resampling
│   │   ├── filterbank.py             #   Gammatone / Mel filterbanks
│   │   └── augmentation.py           #   Data augmentation
│   ├── data/                         # Data loading
│   │   ├── dataset.py                #   Base TSE dataset
│   │   ├── librimix.py               #   LibriMix dataset adapter
│   │   ├── wham.py                   #   WHAM! dataset adapter
│   │   └── collate.py                #   Batch collation (variable-length)
│   ├── training/                     # Training infrastructure
│   │   ├── trainer.py                #   Training loop
│   │   ├── optimizer.py              #   Optimizer & scheduler factory
│   │   ├── losses.py                 #   SI-SNR, PIT, hybrid losses
│   │   └── callbacks.py              #   Early stopping, checkpointing
│   ├── evaluation/                   # Evaluation
│   │   ├── metrics.py                #   SI-SDRi, PESQ, STOI, DNSMOS
│   │   └── evaluator.py              #   Evaluation loop
│   └── inference/                    # Inference
│       └── pipeline.py               #   End-to-end inference pipeline
│
├── experiments/                      # Runnable scripts
│   ├── train.py                      #   Training entry point
│   ├── evaluate.py                   #   Evaluation entry point
│   └── inference_demo.py             #   Interactive demo
│
├── notebooks/                        # Jupyter notebooks
│   └── demo.ipynb                    #   Quick-start demo
│
├── tests/                            # Unit tests
│   ├── test_encoder.py
│   ├── test_models.py
│   └── test_metrics.py
│
└── checkpoints/                      # Saved model weights (git-ignored)
```

---

## Architecture: Auditory-TSE

```
Input: mixture [B, T]  +  enrollment [B, T_ref]
          │                      │
          ▼                      ▼
   ┌──────────────┐     ┌──────────────────┐
   │ 听觉前端      │     │ 说话人编码器      │
   │ Conv-Encoder │     │ ECAPA-TDNN       │
   │ + Gammatone  │     │ → speaker_emb    │
   └──────┬───────┘     │   [B, D_spk]     │
          │              └────────┬─────────┘
   mixture_feat           │
   [B, T', F]             │
          │               │
          ▼               ▼
   ┌─────────────────────────────────────────┐
   │     Separation Network                   │
   │  ┌───────────────────────────────────┐   │
   │  │  Cross-Attention Mask Estimator   │   │
   │  │  mixture_feat + speaker_emb       │   │
   │  │  → target_speaker_mask [B, T', F] │   │
   │  └───────────────────────────────────┘   │
   │  mixture_feat ⊙ mask → separated_feat    │
   └──────────────────┬──────────────────────┘
                      │
                      ▼
   ┌──────────────────────────┐
   │    音频解码器             │
   │    Transposed Conv       │
   │    → separated [B, T]    │
   └──────────────────────────┘
```

### Component Details

1. **听觉前端 (Auditory Encoder)**: 1D Conv with learnable filters, stride = filter_length // 2. Optionally concatenated with Gammatone filterbank output for biologically-inspired frequency decomposition.

2. **说话人编码器 (Speaker Encoder)**: ECAPA-TDNN architecture. Takes enrollment audio, outputs fixed-dimensional speaker embedding (d-vector). Supports loading VoxCeleb-pretrained weights.

3. **分离网络 (Separation Network)**: Conv-TasNet temporal convolutional network (TCN) with cross-attention FiLM layers that modulate features based on the target speaker embedding. Alternative: SepFormer dual-path transformer.

4. **解码器 (Decoder)**: Transposed 1D Conv, symmetric to encoder, reconstructs waveform from masked features.

---

## Key Commands

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Default training with Auditory-TSE
python experiments/train.py model=auditory_tse data=librimix

# Override specific params
python experiments/train.py model=auditory_tse training.batch_size=8 training.epochs=200

# Use SepFormer instead
python experiments/train.py model=sepformer data=librimix
```

### Evaluation

```bash
python experiments/evaluate.py checkpoint=checkpoints/best.ckpt data=librimix
```

### Inference

```bash
python experiments/inference_demo.py \
  --input mixture.wav \
  --enrollment target_speaker.wav \
  --output separated.wav
```

### Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html   # with coverage
```

### Code Quality

```bash
ruff check src/ tests/ experiments/
ruff format src/ tests/ experiments/
```

---

## Code Conventions

### Mandatory
- **Type hints**: All public functions and methods MUST have complete type annotations
- **Docstrings**: Google-style docstrings for all public APIs
- **pathlib.Path**: Use `pathlib.Path` for all file paths — never raw strings
- **No hardcoded paths**: All paths come from Hydra config or CLI args
- **Logging**: Use `logging.getLogger(__name__)` — never `print()` for application output

### Audio Tensor Convention
- Shape: `(batch_size, num_channels, num_samples)` or `(batch_size, num_samples)`
- Sample rate: **16 kHz** default
- Range: float32, typically in [-1.0, 1.0]

### Model Design
- All models inherit from `torch.nn.Module`
- Forward methods accept `**kwargs` for flexibility
- Models should support `torch.jit.script` where possible
- Use `torch.nn.utils.parametrizations` or manual weight norm where needed

---

## Constraints

- **Variable-length audio**: All models must handle variable-length inputs (bucketing + padding)
- **Mixed precision**: Training uses `torch.cuda.amp` autocast
- **Gradient checkpointing**: Enable for large models to save GPU memory
- **Windows compatibility**: All paths use `pathlib.Path`, avoid `/tmp`, use `tempfile` module
- **Checkpointing**: Save top-k checkpoints by validation SI-SDRi; save optimizer state for resume

---

## Key References

- Luo & Mesgarani (2019) — *Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation*
- Subakan et al. (2021) — *Attention is All You Need in Speech Separation* (SepFormer)
- Delcroix et al. (2020) — *Improving Speaker Discrimination of Target Speech Extraction with Time-Domain SpeakerBeam*
- Desplanques et al. (2020) — *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*
- Wang et al. (2023) — *TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation*
