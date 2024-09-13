import cv2
import pytesseract
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

def perform_ocr(image_path):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Perform text extraction
    text = pytesseract.image_to_string(threshold)
    
    return text

def extract_ocr_features(image_dir):
    ocr_features = {}
    
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image_id = os.path.splitext(filename)[0]
            
            ocr_text = perform_ocr(image_path)
            ocr_features[image_id] = ocr_text
    
    return ocr_features

# Extract OCR features for training and test sets
train_ocr_features = extract_ocr_features('preprocessed/train')
test_ocr_features = extract_ocr_features('preprocessed/test')

# Save OCR features
pd.DataFrame.from_dict(train_ocr_features, orient='index', columns=['ocr_text']).to_csv('features/train_ocr_features.csv')
pd.DataFrame.from_dict(test_ocr_features, orient='index', columns=['ocr_text']).to_csv('features/test_ocr_features.csv')

print("OCR feature extraction complete!")
