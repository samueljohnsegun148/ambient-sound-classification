# Copilot Instructions for Ambient Sound Classification

## Project Overview
The Ambient Sound Classification project is designed to classify environmental sounds using machine learning techniques. The project leverages the ESC-50 dataset, which contains 2,000 labeled audio files across 50 categories. The main components of the project include data preprocessing, model training, and inference.

## Architecture
- **Data Flow**: 
  - Audio files are stored in the `data/ESC-50-master/audio/` directory.
  - Metadata and labels are found in `data/ESC-50-master/meta/esc50.csv`.
  - The preprocessing script (`src/preprocess.py`) generates spectrograms from the audio files, which are then used for training the model.

- **Major Components**:
  - `src/preprocess.py`: Handles audio preprocessing and spectrogram generation.
  - `src/train.py`: Trains the model using the processed data.
  - `src/predict.py`: Makes predictions on new audio files.

## Developer Workflows
1. **Setup**:
   - Clone the repository and install dependencies:
     ```bash
     git clone https://github.com/samueljohnsegun148/ambient-sound-classification.git
     cd ambient-sound-classification
     pip install -r requirements.txt
     ```

2. **Dataset Preparation**:
   - Download the dataset using Kaggle CLI or manually, and ensure the directory structure matches:
     ```
     data/
     └── ESC-50-master/
         ├── audio/          # 2,000 .wav files
         └── meta/
             └── esc50.csv   # labels and metadata
     ```

3. **Generating Spectrograms**:
   - Run the preprocessing script:
     ```bash
     python src/preprocess.py
     ```

4. **Training the Model**:
   - Execute the training script:
     ```bash
     python src/train.py
     ```

5. **Making Predictions**:
   - Use the prediction script to classify new audio:
     ```bash
     python src/predict.py path/to/audio.wav
     ```

## Project-Specific Conventions
- The project automatically detects the environment (Kaggle or local) and adjusts paths accordingly, eliminating the need for manual path adjustments.

## Integration Points
- The project relies on external libraries such as `fastai` for model training and `librosa` for audio processing. Ensure these are included in the `requirements.txt`.

## Conclusion
This document serves as a guide for AI agents to navigate the Ambient Sound Classification project effectively. For any unclear sections or additional details needed, please provide feedback for further iterations.
