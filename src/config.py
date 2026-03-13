"""
Configuration and constants for the sound classification project.
"""

import os
from pathlib import Path

# Auto-detect environment
IS_KAGGLE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') != ''

if IS_KAGGLE:
    DATASET_PATH = Path('/kaggle/input/datasets/niranjankn/esc50/ESC-50-master')
    WORKING_DIR = Path('/kaggle/working')
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_PATH = PROJECT_ROOT / 'data' / 'ESC-50-master'
    WORKING_DIR = PROJECT_ROOT / 'output'

# Dataset paths
AUDIO_DIR = DATASET_PATH / 'audio'
META_CSV = DATASET_PATH / 'meta' / 'esc50.csv'

# Output paths
SPECTROGRAM_DIR = WORKING_DIR / 'spectrograms'
MODEL_PATH = WORKING_DIR / 'sound_classifier.pkl'

# Create output dirs if they don't exist
SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)

# Audio processing
DURATION = 5
N_MELS = 128
FMAX = 8000
FIG_SIZE = (2, 2)
DPI = 100

# Training
VALID_PCT = 0.2
SEED = 42
IMAGE_SIZE = 192
BATCH_SIZE = 32
EPOCHS = 15