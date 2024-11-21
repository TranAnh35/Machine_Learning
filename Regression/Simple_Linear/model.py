import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.bias = 0
        
        x_avg = np.mean(X)
        y_avg = np.mean(y)
        
        SS_xy = 0
        SS_x = 0
        
        for i in range(n_samples):
            SS_xy += (X[i][0] - x_avg) * (y[i] - y_avg)
            SS_x += (X[i][0] - x_avg) ** 2
            self.bias += self.lr * (y[i] - y_avg)
            
        B_1 = SS_xy / SS_x
        B_0 = y_avg - B_1 * x_avg
        
        self.weights = [B_0, B_1, self.bias]
            
    def predict(self, X):
        y_approximated = self.bias + self.weights[0] + self.weights[1] * X
        return y_approximated
    
    def mean_squared_error(self, y_true, y_predicted):
        return np.mean((y_true-y_predicted)**2)