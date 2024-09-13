import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ProductDataset(Dataset):
    def __init__(self, ocr_features, cnn_features, labels):
        self.ocr_features = ocr_features
        self.cnn_features = cnn_features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'ocr': self.ocr_features[idx],
            'cnn': self.cnn_features[idx],
            'label': self.labels[idx]
        }

class HybridModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, cnn_feature_dim, num_classes):
        super(HybridModel, self).__init__()
        
        # OCR branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # CNN branch
        self.fc_cnn = nn.Linear(cnn_feature_dim, hidden_dim)
        
        # Combined layers
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, ocr, cnn):
        # OCR branch
        ocr_emb = self.embedding(ocr)
        ocr_out, _ = self.lstm(ocr_emb)
        ocr_out = ocr_out[:, -1, :]  # Take the last output
        
        # CNN branch
        cnn_out = self.fc_cnn(cnn)
        
        # Combine and predict
        combined = torch.cat((ocr_out, cnn_out), dim=1)
        output = self.fc_combined(combined)
        
        return output

# Load features and labels
train_ocr = pd.read_csv('features/train_ocr_features.csv', index_col=0)
train_cnn = pd.read_csv('features/train_cnn_features.csv', index_col=0)
train_labels = pd.read_csv('preprocessed/train_labels.csv')

# Prepare data
le = LabelEncoder()
train_labels['encoded_value'] = le.fit_transform(train_labels['value'])

# Split data
train_data, val_data, train_labels, val_labels = train_test_split(
    pd.concat([train_ocr, train_cnn], axis=1),
    train_labels['encoded_value'],
    test_size=0.2,
    random_state=42
)

# Create datasets and dataloaders
train_dataset = ProductDataset(train_data['ocr_text'].values, train_data.drop('ocr_text', axis=1).values, train_labels.values)
val_dataset = ProductDataset(val_data['ocr_text'].values, val_data.drop('ocr_text', axis=1).values, val_labels.values)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model
vocab_size = 10000  # Adjust based on your vocabulary
embedding_dim = 100
hidden_dim = 128
cnn_feature_dim = train_cnn.shape[1]
num_classes = len(le.classes_)

model = HybridModel(vocab_size, embedding_dim, hidden_dim, cnn_feature_dim, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch['ocr'], batch['cnn'])
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['ocr'], batch['cnn'])
            loss = criterion(outputs, batch['label'])
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch['label'].size(0)
            correct += predicted.eq(batch['label']).sum().item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {loss.item():.4f}, '
          f'Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Accuracy: {100.*correct/total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model training complete and saved!")
