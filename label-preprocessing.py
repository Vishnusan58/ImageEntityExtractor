import pandas as pd
import re
from src.constants import ALLOWED_UNITS

def preprocess_labels(df):
    def extract_value_and_unit(entity_value):
        match = re.match(r'(\d+(?:\.\d+)?)\s*(\w+)', str(entity_value))
        if match:
            value, unit = match.groups()
            return float(value), unit.lower()
        return None, None

    def normalize_unit(unit, entity_name):
        allowed_units = ALLOWED_UNITS.get(entity_name, [])
        if unit in allowed_units:
            return unit
        # Here you might want to add unit conversion logic
        # For example, converting 'gram' to 'kilogram' if needed
        return None

    # Extract value and unit
    df['value'], df['unit'] = zip(*df['entity_value'].map(extract_value_and_unit))
    
    # Normalize units
    df['normalized_unit'] = df.apply(lambda row: normalize_unit(row['unit'], row['entity_name']), axis=1)
    
    # Remove rows with invalid units
    df = df.dropna(subset=['normalized_unit'])
    
    return df

# Load the training data
train_df = pd.read_csv('dataset/train.csv')

# Preprocess the labels
preprocessed_train_df = preprocess_labels(train_df)

# Save the preprocessed data
preprocessed_train_df.to_csv('preprocessed/train_labels.csv', index=False)

print("Label preprocessing complete!")
