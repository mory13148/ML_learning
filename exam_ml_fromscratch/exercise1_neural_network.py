"""
Exercise 1: Neural Network Classification — Iris Dataset
========================================================
Student Name: Mory Adama Dembele
Student ID: _______________________

INSTRUCTIONS:
- Implement all functions marked with TODO
- You may only use NumPy (no sklearn except for data loading)
- Run this file to train your network and see test accuracy
- Target: ≥ 90% test accuracy

"""

import numpy as np
from sklearn.datasets import load_iris  # Only allowed for loading data


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (PROVIDED - DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════

def get_data(test_ratio=0.2, seed=42):
    """
    Load and split the Iris dataset.
    Returns: X_train, X_test, y_train, y_test
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Shuffle
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    
    # Split
    n = int(len(X) * (1 - test_ratio))
    return X[:n], X[n:], y[:n], y[n:]


# ═══════════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

class ReLU:
    """
    ReLU activation function.
    Forward: f(z) = max(0, z)
    Backward: df/dz = 1 if z > 0, else 0
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, z):
        """
        TODO: Implement ReLU forward pass
        
        Args:
            z: Input array of any shape
        
        Returns:
            Output after applying ReLU
        
        Hint: Use np.maximum()
        """
        # TODO: Your code here
        self.cache = z
        return np.max(0, z)
    
    def backward(self, dout):
        """
        TODO: Implement ReLU backward pass
        
        Args:
            dout: Gradient flowing back from the next layer
        
        Returns:
            Gradient with respect to input z
        
        Hint: Gradient is 1 where input was > 0, else 0
        """
        # TODO: Your code here
        return (self.cache > 1) * dout


class Softmax:
    """
    Softmax activation for multi-class classification.
    Forward: softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, z):
        """
        TODO: Implement numerically stable Softmax
        
        Args:
            z: Input array of shape (batch_size, num_classes)
        
        Returns:
            Probabilities of shape (batch_size, num_classes)
        
        Hint: Subtract max(z) before exp() to avoid overflow
        """
        # TODO: Your code here
        prob = np.exp(z - np.max(z)) / np.sum(np.exp(z) - np.max(z))
        self.cache = prob
        return prob
    
    def backward(self, dout):
        """
        TODO: Implement Softmax backward pass
        
        This is tricky! When combined with cross-entropy loss,
        the gradient simplifies to: dL/dz = y_pred - y_true
        
        For now, you can implement the simplified version used with CE loss.
        
        Args:
            dout: Gradient from loss (usually y_pred - y_true)
        
        Returns:
            Same as dout (simplified with cross-entropy)
        """
        # TODO: Your code here
        # Hint: When used with CrossEntropy, this is often just 'return dout'
        return dout


# ═══════════════════════════════════════════════════════════════
# DENSE LAYER (FULLY CONNECTED)
# ═══════════════════════════════════════════════════════════════

class Dense:
    """
    Fully connected layer: z = X @ W + b
    """
    
    def __init__(self, input_size, output_size, seed=42):
        """
        TODO: Initialize weights and biases
        
        Args:
            input_size: Number of input features
            output_size: Number of output neurons
            seed: Random seed for reproducibility
        
        Hint: Use Xavier/Glorot initialization for W:
              W ~ Normal(0, sqrt(2 / (input_size + output_size)))
              b ~ zeros
        """
        np.random.seed(seed)
        
        # TODO: Initialize self.W and self.b
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.b = np.zeros(output_size)
    
        # Cache for backward pass
        self.cache = None
        self.dW = None
        self.db = None
    
    def forward(self, X):
        """
        TODO: Implement forward pass
        
        Args:
            X: Input of shape (batch_size, input_size)
        
        Returns:
            Output of shape (batch_size, output_size)
        """
        # TODO: Your code here
        # Don't forget to cache X for the backward pass!
        self.cache = X
        return X @ self.W + self.b
    
    def backward(self, dout):
        """
        TODO: Implement backward pass
        
        Args:
            dout: Gradient from next layer, shape (batch_size, output_size)
        
        Returns:
            dX: Gradient w.r.t. input, shape (batch_size, input_size)
        
        Also compute and store:
            self.dW: Gradient w.r.t. weights
            self.db: Gradient w.r.t. biases
        
        Hint: Use chain rule
            dX = dout @ W^T
            dW = X^T @ dout
            db = sum(dout, axis=0)
        """
        # TODO: Your code here
        X = self.cache
        self.dW = X.T @ dout
        self.db = np.sum(dout, axis=0)
        self.dX = dout @ self.W.T
        return self.dX


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════

class CrossEntropyLoss:
    """
    Cross-entropy loss for multi-class classification.
    L = -1/N * sum(y_true * log(y_pred + epsilon))
    """
    
    def __init__(self):
        self.cache = None
    
    def forward(self, y_pred, y_true):
        """
        TODO: Compute cross-entropy loss
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels as integers, shape (batch_size,)
        
        Returns:
            Scalar loss value
        
        Hint: 
            - Convert y_true to one-hot encoding
            - Clip y_pred to avoid log(0): np.clip(y_pred, 1e-8, 1 - 1e-8)
            - Loss = -mean(sum(y_true_onehot * log(y_pred), axis=1))
        """
        # TODO: Your code here
        y_true_onehot = np.eye(y_pred.shape[1])[y_true] 
        y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8) 
        loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1))
        return loss
        
    
    def backward(self, y_pred, y_true):
        """
        TODO: Compute gradient of loss w.r.t. predictions
        
        When combined with Softmax, gradient simplifies to:
            dL/dz = (y_pred - y_true_onehot) / batch_size
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels as integers, shape (batch_size,)
        
        Returns:
            Gradient, shape (batch_size, num_classes)
        """
        # TODO: Your code here
        y_true_onehot = np.eye(y_pred.shape[1])[y_true] 
        return (y_pred - y_true_onehot) / y_pred.shape[0]


# ═══════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    BONUS: Implement momentum for +3 points
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Args:
            learning_rate: Step size for weight updates
            momentum: Momentum coefficient (0 = no momentum)
        """
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = {}  # For momentum (bonus)
    
    def update(self, layers):
        """
        TODO: Update weights and biases for all layers
        
        Args:
            layers: List of Dense layer objects
        
        Basic SGD update rule:
            W = W - lr * dW
            b = b - lr * db
        
        BONUS - Momentum update:
            v_W = momentum * v_W + dW
            W = W - lr * v_W
            (same for biases)
        """
        # TODO: Your code here
        for i in layers:
            i.W -= self.dW * self.lr
            i.b -= self.db * self.lr
        


# ═══════════════════════════════════════════════════════════════
# NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════

class NeuralNetwork:
    """
    Multi-layer neural network for classification.
    Architecture: Input -> Dense -> ReLU -> Dense -> ReLU -> Dense -> Softmax
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        TODO: Initialize all layers
        
        Suggested architecture:
            - Input (4) -> Dense(64) -> ReLU
            - Dense(64) -> Dense(32) -> ReLU  
            - Dense(32) -> Dense(3) -> Softmax
        
        Args:
            input_size: Number of input features (4 for Iris)
            hidden_sizes: List of hidden layer sizes, e.g., [64, 32]
            output_size: Number of output classes (3 for Iris)
            learning_rate: Learning rate for optimizer
        """
        # TODO: Create layers and store in lists
        # Example structure:
        self.dense1 = Dense(input_size, hidden_sizes[0])
        self.relu1 = ReLU()
        self.dense2 = Dense(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = ReLU()
        self.dense3 = Dense(hidden_sizes[1], output_size)
        self.softmax = Softmax()
        # ...
        
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = SGD(learning_rate=learning_rate)
    
    def forward(self, X):
        """
        TODO: Implement forward pass through all layers
        
        Args:
            X: Input data, shape (batch_size, input_size)
        
        Returns:
            Output probabilities, shape (batch_size, output_size)
        """
        # TODO: Pass X through all layers sequentially
        pass
    
    def backward(self, dout):
        """
        TODO: Implement backward pass through all layers
        
        Args:
            dout: Gradient from loss function
        
        Call backward() on each layer in reverse order
        """
        # TODO: Your code here
        pass
    
    def train_step(self, X, y):
        """
        TODO: Perform one training step
        
        1. Forward pass
        2. Compute loss
        3. Backward pass
        4. Update weights
        
        Args:
            X: Input batch
            y: True labels
        
        Returns:
            Loss value
        """
        # TODO: Your code here
        pass
    
    def predict(self, X):
        """
        TODO: Make predictions
        
        Args:
            X: Input data
        
        Returns:
            Predicted class labels (integers)
        
        Hint: Use argmax on the output probabilities
        """
        # TODO: Your code here
        pass
    
    def accuracy(self, X, y):
        """
        TODO: Compute classification accuracy
        
        Args:
            X: Input data
            y: True labels
        
        Returns:
            Accuracy as a percentage
        """
        # TODO: Your code here
        pass


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def train_network():
    """
    Main training function.
    """
    print("=" * 60)
    print("Exercise 1: Neural Network Classification")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = get_data()
    
    # TODO: Normalize/standardize features (optional but recommended)
    # Hint: X = (X - mean) / std
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y_train))}")
    print()
    
    # TODO: Create network
    # Example: model = NeuralNetwork(input_size=4, hidden_sizes=[64, 32], output_size=3, learning_rate=0.01)
    
    # TODO: Training loop
    # Train for enough epochs to converge (try 500-2000)
    # Print loss every 50 epochs
    
    # Example structure:
    # epochs = 1000
    # for epoch in range(epochs):
    #     loss = model.train_step(X_train, y_train)
    #     
    #     if (epoch + 1) % 50 == 0:
    #         train_acc = model.accuracy(X_train, y_train)
    #         print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Acc: {train_acc:.2f}%")
    
    # TODO: Final evaluation
    # test_acc = model.accuracy(X_test, y_test)
    # print(f"\n{'='*60}")
    # print(f"Final Test Accuracy: {test_acc:.2f}%")
    # print(f"{'='*60}")
    
    pass


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_network()
