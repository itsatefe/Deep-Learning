import numpy as np

# Random initializations
n, p, m = 3, 2, 2  # example sizes
P = 4  # number of training patterns
alpha = 0.2  # learning rate

# Example training patterns (reduced for clarity)
X = np.array([[1, 1, 1], [1, -1, -1]])
T = np.array([[1, 1], [-1, -1]])

# Initial weights and biases
V = np.array([[.70, .18, .38], [.21, .11, .44]])
W = np.array([[0.77, 0.17], [0.86, 0.29]])
b_H = np.array([0.22, 0.57])  # hidden layer biases
b_O = np.array([0.09, 0.81])  # output layer biases

# Activation functions
def sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return 0.5 * (1 - sigmoid(x) ** 2)

# Compute hidden layer input and output
Z_in = np.dot(X, V.T) + b_H
Z = sigmoid(Z_in)

# Compute output layer input and output
Y_in = np.dot(Z, W.T) + b_O
Y = sigmoid(Y_in)

# Compute output layer error and delta
E =  T - Y
delta_O = E * sigmoid_derivative(Y_in)

# Compute hidden layer delta
delta_H = np.dot(delta_O, W) * sigmoid_derivative(Z_in)

# Calculate weight correction terms
Delta_W = alpha * np.dot(Z.T, delta_O)
Delta_b_O = alpha * np.sum(delta_O, axis=0)
Delta_V = alpha * np.dot(X.T, delta_H)  # Corrected matrix multiplication
Delta_b_H = alpha * np.sum(delta_H, axis=0)

# Update weights and biases
W += Delta_W.T
b_O += Delta_b_O
V += Delta_V.T
b_H += Delta_b_H

# Output results
print("Updated Weights W:\n", W)
print("Updated Biases b_O:\n", b_O)
print("Updated Weights V:\n", V)
print("Updated Biases b_H:\n", b_H)
