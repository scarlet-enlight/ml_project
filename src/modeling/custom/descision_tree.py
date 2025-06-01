from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self, random_state=42, param_grid=None):
        self.random_state = random_state
        self.param_grid = param_grid or {
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 11),
            'criterion': ['gini', 'entropy']
        }
        self.model = None
        self.best_params_ = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            return False
        else:
            return self.model.predict(X)