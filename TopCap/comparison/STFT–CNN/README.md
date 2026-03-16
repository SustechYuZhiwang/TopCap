# STFT-based speech classification models

The difference between the STFT–CNN-8 and STFT–CNN-16 models is that the spectrogram size of the former is 8×8 while that of the latter is 16×16.  The corresponding code goes as follows: 
  ```
  spectrogram = tf.image.resize(spectrogram, [16, 16]) 
  ```

[`STFT–CNN-8.py`](STFT–CNN-8.py) uses the entire dataset for CNN and generates plots using t-SNE and UMAP.  Regarding the plotting, to ensure the data quantity is consistent between the TopCap and STFT–CNN models, a random selection of 1/10 of the dataset was used with TopCap, since STFT–CNN uses 1/10 of the whole dataset as a validation set.  

[`STFT–CNN-16.py`](STFT–CNN-16.py) uses the sampled dataset solely for CNN, catering to the needs of the post-sampling phase of large datasets.  

# STFT-CNN Audio Classification Models

## Project Overview

This repository contains two implementations of a STFT-based Convolutional Neural Network for audio classification. The models share an identical CNN architecture but differ in their input spectrogram resolution and the inclusion of dimensionality reduction visualizations.

## Data Preprocessing

Both models convert raw audio waveforms to spectrograms using short-time Fourier transform (STFT) with frame_length=255 and frame_step=128.

## STFT-CNN-8

STFT-CNN-8.py realises this model as follows.

### Data Processing

- CSV Feature Processing (Lines 1-55):

  - Loads PD_TIMIT.csv (first 2 columns as features, 3rd column as labels 1/2)

  - Performs balanced sampling (2123 samples per class) using scikit-learn's resample

  - Applies t-SNE and UMAP dimensionality reduction to the 2D feature data

  - Saves visualizations as V11.pdf (t-SNE) and V13.pdf (UMAP)

- Audio Dataset Processing (Lines 57-124):

  - Loads raw audio from TIMIT_2classes/TIMIT_2classes/ using TensorFlow's audio_dataset_from_directory

  - Splits data into train/val/test sets (80/10/10)

  - Converts audio to spectrograms and resizes them to 8×8 resolution

### CNN Architecture & Training

- Model Architecture (Lines 125-151):

  - Input: 8×8×1 spectrogram

  - Conv2D(64,3)+ReLU → MaxPool → Conv2D(128,3)+ReLU → MaxPool → Conv2D(256,3)+ReLU → MaxPool

  - Flatten → Dense(64, ReLU) → Dense(1, Sigmoid)

  - Binary cross-entropy loss with Adam optimizer

- Training (Lines 152-161):

  - 10 training epochs with validation monitoring

  - Caching and prefetching for performance optimization

### Latent Space Visualization

- Feature Extraction (Lines 162-201):

  - Extracts features from the last convolutional layer (layer 7)

  - Applies t-SNE and UMAP to the CNN's learned representations

  - Saves visualizations as V12.pdf (t-SNE) and V14.pdf (UMAP)

## STFT-CNN-16

STFT-CNN-16.py realises this model as follows.

### Data Processing

- Audio Sampling (Lines 1-40):

  - Loads audio files from train500_audio_segment_wav_all/

  - Randomly shuffles and selects 2000 samples total

  - Splits into train (1600), validation (200), and test (200) sets

- Spectrogram Generation (Lines 41-80):

  - Converts audio to spectrograms using identical STFT parameters as STFT-CNN-8

  - Resizes spectrograms to 16×16 resolution

  - Caches and prefetches datasets for optimal performance

### CNN Architecture & Training

- Model Architecture (Lines 81-120):

  - Identical architecture to STFT-CNN-8 but with 16×16 input

  - Same layer configuration: Conv2D(64)→Conv2D(128)→Conv2D(256)→Dense(64)→Dense(1)

- Training (Lines 121-130):

  - 10 training epochs with Adam optimizer

  - Binary classification with sigmoid output

  - Note: This model does not include the dimensionality reduction visualization steps present in STFT-CNN-8

## Requirements

Running the above codes requires:

- Python version: Python 3.x

- Key libraries:

  - TensorFlow 2.10+

  - NumPy, Matplotlib, pandas, scikit-learn

  - (STFT-CNN-8 only) umap-learn

Install the required packages using pip:
"""
pip install tensorflow numpy matplotlib pandas scikit-learn umap-learn
"""
