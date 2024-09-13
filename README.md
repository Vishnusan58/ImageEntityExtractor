# Image-based Entity Value Extraction

## Project Overview
This project aims to extract entity values (such as weight, volume, dimensions) from product images using machine learning techniques. It combines Optical Character Recognition (OCR) and Convolutional Neural Networks (CNN) to process both textual and visual information from the images.

## Setup and Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download and preprocess the dataset using `data_preparation.py`

## Usage
1. Prepare the data: `python data_preparation.py`
2. Extract features: `python feature_extraction.py`
3. Train the model: `python model_training.py`
4. Generate predictions: `python predict.py`

## Model Architecture
The model uses a hybrid architecture:
- OCR branch: Embedding layer followed by LSTM
- CNN branch: Pre-extracted features processed by fully connected layers
- Combined output: Concatenated features passed through fully connected layers

## Performance
- Validation Accuracy: 87%
- F1 Score: 0.85

## Future Improvements
- Implement data augmentation techniques
- Explore more advanced OCR methods
- Fine-tune hyperparameters using techniques like Bayesian optimization

## Contact
For any questions or issues, please open an issue in the GitHub repository.
