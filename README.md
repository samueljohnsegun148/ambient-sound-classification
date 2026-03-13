# 🔊 Ambient Sound Classification with Fine-Tuned ResNet18

Fine-tuned a **ResNet18** model to classify **50 categories** of environmental/ambient sounds using mel spectrogram representations. Built on the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50).

## Overview

| | |
|---|---|
| **Task** | Multi-class environmental sound classification (50 classes) |
| **Model** | ResNet18 (pretrained on ImageNet, fine-tuned) |
| **Input** | Raw `.wav` audio → Mel spectrogram images |
| **Framework** | fastai / PyTorch |
| **Dataset** | [ESC-50](https://github.com/karolpiczak/ESC-50) — 2,000 clips across 50 categories |

## Approach

1. **Audio Preprocessing**: Convert `.wav` files to mel spectrograms using `librosa` (128 mel bands, 5s duration)
2. **Dataset Preparation**: Organize spectrograms into category folders, validate images
3. **Transfer Learning**: Fine-tune ImageNet-pretrained ResNet18 with modified classification head
4. **Training**: 15 epochs, batch size 32, 80/20 train/validation split

## Project Structure

```
├── src/
│   ├── config.py       # Auto-detects Kaggle vs local paths
│   ├── preprocess.py   # Audio → spectrogram conversion
│   ├── dataset.py      # DataLoader setup
│   ├── train.py        # Model training
│   └── predict.py      # Inference on new audio
├── notebooks/
│   └── training.ipynb  # Original Kaggle notebook
├── data/               # Dataset (git-ignored, user downloads)
├── output/             # Spectrograms & model (auto-created)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/samueljohnsegun148/ambient-sound-classification.git
cd ambient-sound-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

**Option A — Kaggle CLI:**

```bash
pip install kaggle
kaggle datasets download -d mmoreaux/environmental-sound-classification-50
unzip environmental-sound-classification-50.zip -d data/
```

**Option B — Manual download:**

1. Go to [ESC-50 on Kaggle](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50)
2. Click **Download**
3. Extract into `data/ESC-50-master/`

Your `data/` folder should look like this:

```
data/
└── ESC-50-master/
    ├── audio/          # 2,000 .wav files
    └── meta/
        └── esc50.csv   # labels and metadata
```

### 4. Generate spectrograms

```bash
python src/preprocess.py
```

### 5. Train the model

```bash
python src/train.py
```

### 6. Predict on new audio

```bash
python src/predict.py path/to/audio.wav
```

> **Note:** The project auto-detects whether it's running on Kaggle or locally — no path editing needed.

## Sound Categories (50 classes)

The ESC-50 dataset covers 5 major groups:

- **Animals**: dog, cat, crow, frog, chirping birds, etc.
- **Natural**: rain, sea waves, thunderstorm, wind, water drops
- **Human (non-speech)**: crying baby, sneezing, clapping, breathing, coughing
- **Domestic**: clock tick, door knock, vacuum cleaner, washing machine
- **Urban**: helicopter, siren, car horn, engine, train

## Dataset

This project uses the [ESC-50: Environmental Sound Classification](https://github.com/karolpiczak/ESC-50) dataset.
Available on [Kaggle](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50).

## Acknowledgments

- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) by Karol J. Piczak
- [fastai](https://docs.fast.ai/) for transfer learning framework
- [librosa](https://librosa.org/) for audio processing