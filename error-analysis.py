import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load validation data and predictions
val_data = pd.read_csv('validation_data.csv')
val_predictions = pd.read_csv('validation_predictions.csv')

# Compute classification report
print(classification_report(val_data['true_label'], val_predictions['predicted_label']))

# Compute confusion matrix
cm = confusion_matrix(val_data['true_label'], val_predictions['predicted_label'])

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Analyze errors
errors = val_data[val_data['true_label'] != val_predictions['predicted_label']]

# Print some example errors
print("Example Errors:")
for _, row in errors.head().iterrows():
    print(f"True: {row['true_label']}, Predicted: {row['predicted_label']}, Image: {row['image_link']}")

# Analyze error distribution by entity type
error_by_entity = errors['entity_name'].value_counts(normalize=True)
plt.figure(figsize=(10, 6))
error_by_entity.plot(kind='bar')
plt.title('Error Distribution by Entity Type')
plt.ylabel('Error Rate')
plt.xlabel('Entity Type')
plt.savefig('error_distribution.png')
plt.close()

print("Error analysis complete. Check 'confusion_matrix.png' and 'error_distribution.png' for visualizations.")
