import numpy as np

class MultipleLinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Tính toán ma trận nghịch đảo
        X_transpose = X.T
        X_transpose_X = X_transpose @ X
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        
        # Tính toán hệ số hồi quy
        self.w = X_transpose_X_inv @ X_transpose @ y
        

    def predict(self, X):
        X = np.array(X)
        return X @ self.w

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)