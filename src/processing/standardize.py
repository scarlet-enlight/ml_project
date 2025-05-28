import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def standardize(data):
    X = np.array(data)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_DIR / "Sleep_health_and_lifestyle_dataset.csv")
    numeric_data = df.select_dtypes(include=[np.number])
    scaled_data = standardize(numeric_data)
    print("Standardized data:")
    print(scaled_data)