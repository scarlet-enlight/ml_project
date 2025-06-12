from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class DecisionTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.is_trained = False
        print("Rzeczywista głębokość drzewa:", model.get_depth())
        print("Liczba liści:", model.get_n_leaves())

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None.")
        self.model.fit(self.X, self.y)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        if isinstance(X, (list, tuple)) or X.ndim == 1:
            X = pd.DataFrame([X], columns=self.X.columns)
        elif isinstance(X, pd.DataFrame):
            X = X[self.X.columns]
        else:
            X = pd.DataFrame(X, columns=self.X.columns)
        return self.model.predict(X)
