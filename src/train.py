"""
Model training — fine-tunes a pretrained ResNet18 on spectrogram images.
"""

from fastai.vision.all import vision_learner, resnet18, error_rate
from fastai.callback.progress import ProgressCallback

from config import MODEL_PATH, EPOCHS
from dataset import create_dataloaders


def train():
    """Fine-tune ResNet18 for ambient sound classification."""
    print("Creating dataloaders...")
    dls = create_dataloaders()

    print(f"Training with {len(dls.vocab)} categories: {list(dls.vocab)}")
    print(f"Fine-tuning ResNet18 for {EPOCHS} epochs...\n")

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.remove_cb(ProgressCallback)
    learn.fine_tune(EPOCHS)

    # Export trained model
    learn.export(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    return learn


if __name__ == '__main__':
    train()