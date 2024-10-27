# Age Prediction from Audio Features

This project uses a linear regression model to predict age from various audio features extracted from speech data. The model uses features such as pitch, formants, intensity, spectral properties, MFCCs, and zero crossing rate to generate predictions.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

## Overview
The goal of this project is to predict the age of speakers based on audio features extracted from their speech recordings. The features are used to train a linear regression model, which can generalize age prediction across different age groups, ranging from teens to eighties.

## Installation

To run this project, you'll need to install the following dependencies:

```bash
pip install numpy pandas matplotlib seaborn librosa pydub parselmouth
```

Ensure you have Python 3.7+ installed. You will also need the necessary audio processing libraries and modules for feature extraction.

## Dataset
The dataset contains audio files and corresponding age labels. The dataset is structured in CSV format, containing:
- `filename`: the name of the audio file.
- `age`: the age group of the speaker.

You should have the following CSV files:
- `truncated_train.csv`: training data with filenames and age groups.
- `cv-valid-test.csv`: test data with filenames and age groups.

## Feature Extraction

The project extracts the following audio features from the dataset using `librosa` and `parselmouth`:
- **Pitch**: Average pitch of the audio file.
- **Formants**: Formant frequencies extracted using Burg’s method.
- **Intensity**: Average root mean square (RMS) intensity.
- **Duration**: Duration of the audio.
- **Spectral Centroid**: Weighted mean of the frequencies in the signal.
- **Spectral Bandwidth**: Measure of the width of the spectrum.
- **Spectral Contrast**: Difference in amplitude between peaks and valleys in the spectrum.
- **Spectral Flatness**: Degree to which the spectrum is flat.
- **MFCC**: Mel-frequency cepstral coefficients.
- **Chroma**: Energy distribution across the 12 pitch classes.
- **Zero Crossing Rate**: Rate of sign-changes of the signal over time.

The extracted features are saved into CSV files:
- `Features.csv`: Contains extracted features from the training data.
- `Test_Feature.csv`: Contains extracted features from the test data.

## Model Training

This project implements a custom-built multivariate linear regression model from scratch, without using any external machine learning libraries. The model uses gradient descent to optimize weights for each feature and the intercept term, minimizing the mean squared error (MSE) between the predicted and actual ages.

### Parameters
- **Learning Rate (`lr`)**: The rate at which the model learns (default set to 0.00000001).
- **Epochs**: Number of iterations over the entire training set (default set to 10,000).

### Features Selection
The model selects the top 4 features with the strongest correlation with age for further analysis.

## Evaluation

The model's performance is evaluated based on the accuracy of predicted ages compared to the actual ages in the test dataset. The accuracy is calculated as the percentage of correct predictions.

## Usage

To use this project, open the Jupyter Notebook (`age_prediction.ipynb`) and run the cells sequentially to execute the code, extract features, train the model, and evaluate its performance. 

The notebook demonstrates the implementation of multivariate linear regression from scratch, detailing the calculation of gradients and updating coefficients manually, offering insights into how linear regression works at a fundamental level.

## Results

The accuracy of the model is printed at the end of the training. Additionally, the KDE plots visualize the distribution of each feature across different age groups.

## Contributing
Feel free to contribute to this project by creating pull requests, reporting issues, or suggesting new features.
