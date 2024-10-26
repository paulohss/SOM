import numpy as np
import pandas as pd
import yaml

def load_data_from_csv(file_path, columns):
    """
    Load specified columns from a CSV file as input data for SOM.

    Parameters:
    - file_path: str
        Path to the CSV file.
    - columns: list of str
        List of columns to use as input features.

    Returns:
    - np.ndarray
        Data from specified columns, normalized for training.
    """
    # Load CSV data, selecting specified columns.
    data = pd.read_csv(file_path, usecols=columns)
    
    # Validate that the CSV contains exactly 3 columns
    if data.shape[1] != 3:
        raise ValueError("CSV file must contain exactly 3 columns. Please ensure your data file meets this requirement.")
    
  # Ensure the specified features exist in the file
    missing_features = [feature for feature in columns if feature not in data.columns]
    if missing_features:
        raise KeyError(f"The following features were not found in the file: {', '.join(missing_features)}")    

    # Normalize data to [0, 1] range.
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data.values

def load_config(config_file='config.yaml'):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    - config_file: str
        Path to the YAML configuration file.

    Returns:
    - dict
        Dictionary containing configuration parameters.
    """
    # Load YAML configuration.
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
