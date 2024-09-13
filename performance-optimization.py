import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time

def time_inference(model, dataloader):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            _ = model(batch['ocr'], batch['cnn'])
    end_time = time.time()
    return end_time - start_time

# Load the model
model = HybridModel(vocab_size, embedding_dim, hidden_dim, cnn_feature_dim, num_classes)
model.load_state_dict(torch.load('model.pth'))

# Create a sample dataloader
sample_dataset = ProductDataset(...)  # Fill with sample data
sample_dataloader = DataLoader(sample_dataset, batch_size=32)

# Time the original model
original_time = time_inference(model, sample_dataloader)
print(f"Original inference time: {original_time:.4f} seconds")

# Optimize the model
optimized_model = torch.jit.script(model)

# Time the optimized model
optimized_time = time_inference(optimized_model, sample_dataloader)
print(f"Optimized inference time: {optimized_time:.4f} seconds")

# Save the optimized model
torch.jit.save(optimized_model, 'optimized_model.pth')

print("Performance optimization complete. Optimized model saved as 'optimized_model.pth'")
