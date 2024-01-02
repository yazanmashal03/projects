import numpy as np

# Function to compute the least squares criterion
def least_squares(X, y, beta):
    return 0.5 * np.linalg.norm(y - X @ beta) ** 2

# Gradient of the least squares criterion
def gradient_least_squares(X, y, beta):
    return X.T @ (X @ beta - y)

# Gradient Descent for Linear Regression
def gradient_descent_linear_regression(X, y, alpha, num_iterations):
    # Initial guess for beta
    beta = np.zeros(X.shape[1])
    # Record the progression of the cost function value
    cost_history = []

    for i in range(num_iterations):
        # Calculate the gradient
        gradient = gradient_least_squares(X, y, beta)
        # Update beta
        beta -= alpha * gradient
        # Calculate the cost for the current iteration
        cost = least_squares(X, y, beta)
        cost_history.append(cost)
        print(f"Iteration {i+1}: beta = {beta}, cost = {cost}")

    return beta, cost_history

def gradient_descent_linear_regression_with_dropout(X, y, alpha, num_iterations, dropout_rate):
    np.random.seed(42)  # For reproducibility
    beta = np.zeros(X.shape[1])
    cost_history = []

    for i in range(num_iterations):
        # Simulate dropout: create a dropout mask that randomly sets some weights to 0
        dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=beta.shape)
        beta_dropped = beta * dropout_mask

        # Calculate the gradient with the dropped-out beta
        gradient = gradient_least_squares(X, y, beta_dropped)
        
        # Apply the dropout mask to the gradient as well
        gradient *= dropout_mask

        # Update beta using the masked gradient
        beta -= alpha * gradient

        # Calculate the cost with the non-dropped-out beta for evaluation
        cost = least_squares(X, y, beta)
        cost_history.append(cost)

        print(f"Iteration {i+1}: beta = {beta}, cost = {cost}")

    return beta, cost_history

# Example usage: We'll create a random X and y to demonstrate the usage of the function
np.random.seed(42) # Seed for reproducibility
n_samples = 100
n_features = 3

# Random design matrix X and outcomes Y
X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples)

# Learning rate (alpha) and number of iterations
alpha = 0.01
num_iterations = 50
dropout_rate = 0.2

# Run gradient descent
beta_optimal, cost_history = gradient_descent_linear_regression_with_dropout(X, y, alpha, num_iterations, dropout_rate)

# The optimal beta and the history of cost function value
beta_optimal, cost_history[-1]
