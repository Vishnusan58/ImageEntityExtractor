import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.constants import ALLOWED_UNITS
from src.sanity import run_sanity_check

# Load the trained model
model = HybridModel(vocab_size, embedding_dim, hidden_dim, cnn_feature_dim, num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load test data
test_ocr = pd.read_csv('features/test_ocr_features.csv', index_col=0)
test_cnn = pd.read_csv('features/test_cnn_features.csv', index_col=0)
test_data = pd.concat([test_ocr, test_cnn], axis=1)

# Load the original test file to get entity_name
original_test = pd.read_csv('dataset/test.csv')

# Create test dataset
test_dataset = ProductDataset(test_data['ocr_text'].values, test_data.drop('ocr_text', axis=1).values, np.zeros(len(test_data)))
test_loader = DataLoader(test_dataset, batch_size=32)

# Make predictions
predictions = []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch['ocr'], batch['cnn'])
        _, predicted = outputs.max(1)
        predictions.extend(predicted.cpu().numpy())

# Inverse transform predictions
le = LabelEncoder()
le.classes_ = np.load('label_encoder_classes.npy')  # Load the classes from training
predicted_values = le.inverse_transform(predictions)

# Post-process predictions
def format_prediction(value, entity_name):
    allowed_units = ALLOWED_UNITS.get(entity_name, [])
    if not allowed_units:
        return ""
    
    # Choose the most appropriate unit (this is a simplified approach)
    unit = allowed_units[0]
    
    # Format the output string
    return f"{value:.2f} {unit}"

# Generate output dataframe
output_df = pd.DataFrame({
    'index': original_test['index'],
    'prediction': [format_prediction(value, entity_name) 
                   for value, entity_name in zip(predicted_values, original_test['entity_name'])]
})

# Save the output file
output_file = 'test_out.csv'
output_df.to_csv(output_file, index=False)
print(f"Output file '{output_file}' generated.")

# Run sanity check
print("Running sanity check...")
run_sanity_check(output_file)
print("Sanity check complete.")
