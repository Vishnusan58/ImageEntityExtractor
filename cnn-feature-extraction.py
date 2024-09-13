import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

# Load pre-trained ResNet model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_cnn_features(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        features = model(img_tensor)
    
    return features.squeeze().numpy()

def extract_cnn_features_batch(image_dir):
    cnn_features = {}
    
    for filename in tqdm(os.listdir(image_dir)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            image_id = os.path.splitext(filename)[0]
            
            features = extract_cnn_features(image_path)
            cnn_features[image_id] = features
    
    return cnn_features

# Extract CNN features for training and test sets
train_cnn_features = extract_cnn_features_batch('preprocessed/train')
test_cnn_features = extract_cnn_features_batch('preprocessed/test')

# Save CNN features
pd.DataFrame.from_dict(train_cnn_features, orient='index').to_csv('features/train_cnn_features.csv')
pd.DataFrame.from_dict(test_cnn_features, orient='index').to_csv('features/test_cnn_features.csv')

print("CNN feature extraction complete!")
