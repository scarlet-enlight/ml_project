from math import sqrt, pi, exp, log

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class NaiveBayes:
    def __init__(self):
        self.classes = []
        self.priors = []
        self.params = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {c: np.mean(y == c) for c in self.classes}
        self.params = {
            c: {
                'mean': X[y == c].mean(axis=0),
                'std': X[y == c].std(axis=0),
            }
            for c in self.classes
        }

    def gauss(self, x, mu, sigma, eps=1e-9):
        sigma = np.clip(sigma, eps, None)
        return (1 / (sqrt(2 * pi) * sigma)) * exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []
        for x in X:
            best_class, best_logp = None, -np.inf
            for c in self.classes:
                logp = log(self.priors[c])
                for xi, m, s in zip(x, self.params[c]['mean'], self.params[c]['std']):
                    p = self.gauss(xi, m, s)
                    logp += log(p + 1e-9)
                if logp > best_logp:
                    best_logp, best_class = logp, c
            preds.append(best_class)
        return np.array(preds)
