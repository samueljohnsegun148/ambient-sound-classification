"""
Inference — predict sound category from an audio file.
"""

from fastai.vision.all import load_learner, PILImage

from config import MODEL_PATH, WORKING_DIR
from preprocess import audio_to_spectrogram


def predict(audio_path, model_path=MODEL_PATH):
    """Predict the category of a sound from its audio file."""
    # Convert audio to spectrogram
    spectrogram_path = WORKING_DIR / 'temp_predict.png'
    audio_to_spectrogram(audio_path, spectrogram_path)

    # Load model and predict
    learn = load_learner(model_path)
    label, _, probs = learn.predict(PILImage.create(spectrogram_path))

    print(f"Predicted: {label}\n")
    print("All probabilities:")
    for i, category in enumerate(learn.dls.vocab):
        print(f"  {category}: {probs[i]:.4f}")

    return label, probs


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
    else:
        predict(sys.argv[1])