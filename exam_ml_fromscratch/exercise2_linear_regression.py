"""
Exercise 2: Linear Regression — Greenfield Estates
====================================================
Student Name: _______________________
Student ID: _______________________

INSTRUCTIONS:
- Implement all functions marked with TODO
- You may only use NumPy
- Compare Normal Equation vs Gradient Descent results
- Predict prices for the 3 test samples

"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# DATASET (PROVIDED - DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════════

def load_housing_data():
    """
    Load the Greenfield Estates housing dataset.
    
    Features:
        0: Size (m²)
        1: Age (years)
        2: Lake Distance (km)
        3: Solar Panels (count)
    
    Target: Price (k€)
    
    Returns:
        X_train: Training features (15 samples x 4 features)
        y_train: Training prices (15 samples)
        X_test: Test features (3 samples x 4 features)
    """
    # Training data (15 samples)
    X_train = np.array([
        [120, 5,  1.2, 8],   # Sample 1
        [85,  12, 3.5, 4],   # Sample 2
        [200, 2,  0.8, 12],  # Sample 3
        [150, 8,  2.1, 6],   # Sample 4
        [95,  15, 4.0, 3],   # Sample 5
        [180, 3,  1.0, 10],  # Sample 6
        [60,  20, 5.5, 2],   # Sample 7
        [140, 6,  1.8, 7],   # Sample 8
        [110, 10, 2.8, 5],   # Sample 9
        [170, 4,  1.5, 9],   # Sample 10
        [75,  18, 4.8, 2],   # Sample 11
        [160, 7,  2.0, 8],   # Sample 12
        [130, 9,  3.0, 5],   # Sample 13
        [190, 1,  0.6, 14],  # Sample 14
        [100, 14, 3.8, 4],   # Sample 15
    ], dtype=float)
    
    y_train = np.array([
        310, 190, 520, 380, 170, 470, 110, 350, 250, 430, 140, 400, 280, 550, 200
    ], dtype=float)
    
    # Test data (3 samples) - prices unknown
    X_test = np.array([
        [145, 7,  1.9, 7],   # Sample 16
        [88,  13, 3.2, 3],   # Sample 17
        [210, 2,  0.5, 15],  # Sample 18
    ], dtype=float)
    
    return X_train, y_train, X_test


# ═══════════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def standardize_features(X_train, X_test):
    """
    TODO: Standardize features using z-score normalization.
    
    CRITICAL: Use ONLY training set statistics (mean and std).
    Apply the same transformation to the test set.
    
    Formula: X_norm = (X - mean) / std
    
    Args:
        X_train: Training features, shape (n_train, n_features)
        X_test: Test features, shape (n_test, n_features)
    
    Returns:
        X_train_norm: Normalized training features
        X_test_norm: Normalized test features
        mean: Training set mean (for reference)
        std: Training set std (for reference)
    
    Hint: Use np.mean() and np.std() with axis=0 for column-wise stats
    
    Why does this matter? Explain in a comment below:
    """
    # TODO: Your code here
    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test_norm = (X_test - np.mean(X_test)) /np.std(X_test)
    # QUESTION: Why must we use training statistics for both sets?
    # YOUR ANSWER: 
    
    return X_train_norm, X_test_norm, np.mean(X_train), np.std(X_train)
def add_bias_column(X):
    """
    TODO: Add a column of ones to X for the bias term.
    
    This allows us to write: y = X @ theta
    where theta includes both weights and bias.
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
    
    Returns:
        X_with_bias: Shape (n_samples, n_features + 1)
        First column is all ones.
    
    Hint: Use np.hstack() or np.c_[]
    """
    # TODO: Your code here
    n_samples, n_features = X.shape
    biais = np.ones((n_samples, n_features+1))

    return np.hstack((X, biais))


# ═══════════════════════════════════════════════════════════════
# NORMAL EQUATION (CLOSED-FORM SOLUTION)
# ═══════════════════════════════════════════════════════════════

def normal_equation(X, y):
    """
    TODO: Solve for optimal weights using the Normal Equation.
    
    Formula: θ = (X^T X)^(-1) X^T y
    
    Args:
        X: Feature matrix with bias column, shape (n_samples, n_features + 1)
        y: Target values, shape (n_samples,)
    
    Returns:
        theta: Optimal parameters, shape (n_features + 1,)
    
    Hint: Use np.linalg.inv() for matrix inversion
          or np.linalg.solve() for better numerical stability
    """
    # TODO: Your code here
    XT_X = X.T @ X
    yT_y = X.T @ y

    return np.linalg.pinv(XT_X) @ yT_y


# ═══════════════════════════════════════════════════════════════
# GRADIENT DESCENT
# ═══════════════════════════════════════════════════════════════

def compute_cost(X, y, theta):
    """
    TODO: Compute Mean Squared Error (MSE) loss.
    
    Formula: MSE = (1/N) * sum((y_pred - y_true)^2)
             where y_pred = X @ theta
    
    Args:
        X: Feature matrix with bias, shape (n_samples, n_features + 1)
        y: True values, shape (n_samples,)
        theta: Current parameters, shape (n_features + 1,)
    
    Returns:
        cost: Scalar MSE value
    """
    # TODO: Your code here
    n_samples, _ = X.shape
    y_pred = X @ theta
    erreur = y_pred - y
    return (1 / n_samples) * np.sum(erreur**2)


def compute_gradient(X, y, theta):
    """
    TODO: Compute gradient of MSE with respect to theta.
    
    Formula: ∇MSE = (2/N) * X^T (X @ theta - y)
    
    Args:
        X: Feature matrix with bias, shape (n_samples, n_features + 1)
        y: True values, shape (n_samples,)
        theta: Current parameters, shape (n_features + 1,)
    
    Returns:
        gradient: Shape (n_features + 1,)
    """
    # TODO: Your code here
    n_samples = X[0]
    y_pred = X @ theta
    erreur = y_pred - y
     
    return (2 / n_samples) * (X.T @ erreur)


def gradient_descent(X, y, learning_rate=0.001, n_iterations=1000, print_every=100):
    """
    Optimize theta using gradient descent.
    
    Update rule: θ = θ - α * ∇MSE
    
    Args:
        X: Feature matrix with bias, shape (n_samples, n_features + 1)
        y: Target values, shape (n_samples,)
        learning_rate: Step size (α)
        n_iterations: Number of iterations
        print_every: Print loss every N iterations
    
    Returns:
        theta: Optimized parameters, shape (n_features + 1,)
        cost_history: List of costs at each iteration
    """
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)  
    cost_history = []

    for i in range(n_iterations):
        gradient = compute_gradient(X, y, theta)
        theta -= learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        if i % print_every == 0:
            print(f"Iteration {i} : Cost {cost:.4f}")

    return theta, cost_history


# ═══════════════════════════════════════════════════════════════
# PREDICTION & EVALUATION
# ═══════════════════════════════════════════════════════════════

def predict(X, theta):
    """
    TODO: Make predictions using learned parameters.
    
    Formula: y_pred = X @ theta
    
    Args:
        X: Feature matrix with bias column
        theta: Learned parameters
    
    Returns:
        predictions: Predicted values
    """
    # TODO: Your code here
    return X @ theta


def compute_r2(y_true, y_pred):
    """
    TODO: Compute R² (coefficient of determination).
    
    Formula: R² = 1 - (SS_res / SS_tot)
             SS_res = sum((y_true - y_pred)^2)
             SS_tot = sum((y_true - mean(y_true))^2)
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        r2: R² score (1.0 is perfect, 0.0 is baseline)
    """
    # TODO: Your code here
    SS_res = np.sum((y_true - y_pred)**2)
    SS_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (SS_res / SS_tot)


def compute_rmse(y_true, y_pred):
    """
    TODO: Compute Root Mean Squared Error.
    
    Formula: RMSE = sqrt(MSE) = sqrt((1/N) * sum((y_true - y_pred)^2))
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        rmse: Root mean squared error
    """
    # TODO: Your code here
    n_samples = len(y_true)
    mse = np.sqrt((1 / n_samples) * np.sum((y_true - y_pred)**2))
    return np.sqrt(mse)


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════

def main():
    """
    Main function to train and evaluate both methods.
    """
    print("=" * 70)
    print("Exercise 2: Linear Regression — Greenfield Estates")
    print("=" * 70)
    print()
    
    # Load data
    X_train, y_train, X_test = load_housing_data()
    
    print("Dataset Info:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]} (Size, Age, Lake Distance, Solar Panels)")
    print(f"  Test samples: {X_test.shape[0]}")
    print()
    
    # ──────────────────────────────────────────────────────────────
    # TODO: Step 1 - Standardize features
    # ──────────────────────────────────────────────────────────────
    
    X_train_norm, X_test_norm, train_mean, train_std = standardize_features(X_train, X_test)
    
    print("Standardization complete.")
    print(f"  Training mean: {train_mean}")
    print(f"  Training std: {train_std}")
    print()
    
    # ──────────────────────────────────────────────────────────────
    # TODO: Step 2 - Add bias column
    # ──────────────────────────────────────────────────────────────
    
    X_train_bias = add_bias_column(X_train_norm)
    X_test_bias = add_bias_column(X_test_norm)
    
    # ──────────────────────────────────────────────────────────────
    # TODO: Step 3 - Normal Equation
    # ──────────────────────────────────────────────────────────────
    
    print("-" * 70)
    print("METHOD 1: Normal Equation (Closed-Form Solution)")
    print("-" * 70)
    
    theta_ne = normal_equation(X_train_bias, y_train)
    print(f"Optimal parameters (Normal Eq): {theta_ne}")
    print()
    
    # # Predictions on training set
    y_train_pred_ne = predict(X_train_bias, theta_ne)
    r2_ne = compute_r2(y_train, y_train_pred_ne)
    rmse_ne = compute_rmse(y_train, y_train_pred_ne)
    
    print(f"Training Set Performance:")
    print(f"  R² score: {r2_ne:.4f}")
    print(f"  RMSE: {rmse_ne:.2f} k€")
    print()
    
    # # Predictions on test set
    y_test_pred_ne = predict(X_test_bias, theta_ne)
    print(f"Test Set Predictions (Normal Equation):")
    for i, pred in enumerate(y_test_pred_ne, start=16):
        print(f"  Sample {i}: {pred:.2f} k€")
    print()
    
    # ──────────────────────────────────────────────────────────────
    # TODO: Step 4 - Gradient Descent
    # ──────────────────────────────────────────────────────────────
    
    print("-" * 70)
    print("METHOD 2: Gradient Descent (Iterative Optimization)")
    print("-" * 70)
    
    theta_gd, cost_history = gradient_descent(
        X_train_bias, 
        y_train, 
        learning_rate=0.01,
        n_iterations=1000,
        print_every=100
    )
    
    print()
    print(f"Optimal parameters (Gradient Descent): {theta_gd}")
    print()
    
    # # Predictions on training set
    y_train_pred_gd = predict(X_train_bias, theta_gd)
    r2_gd = compute_r2(y_train, y_train_pred_gd)
    rmse_gd = compute_rmse(y_train, y_train_pred_gd)
    
    print(f"Training Set Performance:")
    print(f"  R² score: {r2_gd:.4f}")
    print(f"  RMSE: {rmse_gd:.2f} k€")
    print()
    
    # # Predictions on test set
    y_test_pred_gd = predict(X_test_bias, theta_gd)
    print(f"Test Set Predictions (Gradient Descent):")
    for i, pred in enumerate(y_test_pred_gd, start=16):
        print(f"  Sample {i}: {pred:.2f} k€")
    print()
    
    # ──────────────────────────────────────────────────────────────
    # TODO: Step 5 - Compare both methods
    # ──────────────────────────────────────────────────────────────
    
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"Parameter difference (L2 norm): {np.linalg.norm(theta_ne - theta_gd):.6f}")
    print()
    print("Do both methods give similar results?")
    print("YOUR ANSWER:")
    print("NO normal equation  > gradient_descent")
    print()
    
    # ──────────────────────────────────────────────────────────────
    # BONUS: Feature Importance
    # ──────────────────────────────────────────────────────────────
    
    print("-" * 70)
    print("BONUS: Feature Importance Analysis")
    print("-" * 70)
    
    # TODO: Which feature has the biggest impact on price?
    # Analyze the learned coefficients (excluding bias term)
    # Remember: features are standardized, so coefficients are comparable
    
    feature_names = ["Size (m²)", "Age (years)", "Lake Dist (km)", "Solar Panels"]
    coefficients = theta_ne[1:]  # Exclude bias
    
    print("Feature coefficients (standardized scale):")
    for name, coef in zip(feature_names, coefficients):
        print(f"  {name:20s}: {coef:+.4f}")
    print()
    
    print("Most important feature:")
    print("YOUR ANSWER:")
    print("Age (years)")
    print()


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
