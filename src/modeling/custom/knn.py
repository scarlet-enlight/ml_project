import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


class KNN:
    def __init__(self, k=2):
        self.k = k

    def fit(self, X_train=None, y_train=None):
        self.X = X_train
        self.y = y_train

    def distance(self, v1, v2):
        return np.linalg.norm(v2 - v1)

    def predict(self, vector):
        distances = [self.distance(vector, row) for idx, row in self.X.iterrows()]
        sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
        neighbors = sorted_distances[:self.k]
        votes = [self.y.iloc[neighbor[0]] for neighbor in neighbors]
        return max(set(votes), key=votes.count)
