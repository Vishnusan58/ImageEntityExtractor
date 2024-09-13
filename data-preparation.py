import pandas as pd
import os
from src.utils import download_images

# Load the CSV files
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# Create directories for images if they don't exist
os.makedirs('images/train', exist_ok=True)
os.makedirs('images/test', exist_ok=True)

# Download training images
for index, row in train_df.iterrows():
    image_url = row['image_link']
    image_filename = f"images/train/{index}.jpg"
    download_images(image_url, image_filename)
    print(f"Downloaded training image {index}")

# Download test images
for index, row in test_df.iterrows():
    image_url = row['image_link']
    image_filename = f"images/test/{index}.jpg"
    download_images(image_url, image_filename)
    print(f"Downloaded test image {index}")

print("Data preparation complete!")
