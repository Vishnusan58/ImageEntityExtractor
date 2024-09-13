import cv2
import numpy as np
import os
from tqdm import tqdm

def preprocess_image(image_path, target_size=(224, 224)):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def preprocess_dataset(image_dir, output_dir, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(image_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Preprocess the image
            preprocessed_img = preprocess_image(input_path, target_size)
            
            # Save the preprocessed image
            cv2.imwrite(output_path, cv2.cvtColor((preprocessed_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# Preprocess training and test datasets
preprocess_dataset('images/train', 'preprocessed/train')
preprocess_dataset('images/test', 'preprocessed/test')

print("Image preprocessing complete!")
