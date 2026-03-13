"""
Audio preprocessing — converts .wav files into mel spectrogram images.
"""

import time
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from pathlib import Path
from fastai.vision.all import verify_images, get_image_files

from config import (
    AUDIO_DIR, META_CSV, SPECTROGRAM_DIR,
    DURATION, N_MELS, FMAX, FIG_SIZE, DPI
)


def audio_to_spectrogram(audio_path, save_path, duration=DURATION):
    """Convert a .wav audio file into a mel spectrogram .png image."""
    y, sr = librosa.load(audio_path, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = plt.axes([0, 0, 1, 1])
    librosa.display.specshow(S_db, ax=ax, fmax=FMAX)
    plt.savefig(save_path, bbox_inches=None)
    plt.close(fig)


def generate_all_spectrograms():
    """Convert all audio files in the ESC-50 dataset to spectrograms."""
    df = pd.read_csv(META_CSV)
    categories = df['category'].unique()

    print(f"Found {len(categories)} categories")
    print(f"Class distribution:\n{df['category'].value_counts()}\n")

    for category in categories:
        dest = SPECTROGRAM_DIR / category
        dest.mkdir(exist_ok=True, parents=True)

        clips = df[df['category'] == category]
        for _, row in clips.iterrows():
            save_path = dest / f"{row['filename'].replace('.wav', '.png')}"
            audio_to_spectrogram(AUDIO_DIR / row['filename'], save_path)

        print(f"{category}: {len(clips)} spectrograms created")
        time.sleep(1)

    # Clean up corrupted images
    failed = verify_images(get_image_files(SPECTROGRAM_DIR))
    failed.map(Path.unlink)
    print(f"\nRemoved {len(failed)} corrupted images")


if __name__ == '__main__':
    generate_all_spectrograms()