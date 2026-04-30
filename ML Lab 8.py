import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# A1: Basic Units
# -------------------------------------------------

def summation_unit(x, w):
    return np.dot(x, w)

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def error_unit(target, output):
    return target - output


# -------------------------------------------------
# A2: Perceptron Training
# -------------------------------------------------

def train_perceptron(X, y, weights, lr, activation):
    errors = []
    epochs = []

    for epoch in range(1000):
        total_error = 0

        for i in range(len(X)):
            net = summation_unit(X[i], weights)
            out = activation(net)
            err = error_unit(y[i], out)

            weights = weights + lr * err * X[i]
            total_error += err ** 2

        errors.append(total_error)
        epochs.append(epoch)

        if total_error <= 0.002:
            break

    return weights, epochs, errors


# -------------------------------------------------
# A3: Compare Activation Functions
# -------------------------------------------------

def compare_activation(X, y, weights, lr, activations):
    results = {}
    for name, func in activations.items():
        _, epochs, _ = train_perceptron(X, y, weights.copy(), lr, func)
        results[name] = len(epochs)
    return results


# -------------------------------------------------
# A4: Learning Rate Analysis
# -------------------------------------------------

def learning_rate_test(X, y, weights, activation):
    lr_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    iterations = []

    for lr in lr_values:
        _, epochs, _ = train_perceptron(X, y, weights.copy(), lr, activation)
        iterations.append(len(epochs))

    return lr_values, iterations


# -------------------------------------------------
# A7: Pseudo-Inverse
# -------------------------------------------------

def pseudo_inverse_solution(X, y):
    return np.linalg.pinv(X).dot(y)


# -------------------------------------------------
# A8: Simple Neural Network (Backprop)
# -------------------------------------------------

def sigmoid_derivative(x):
    return x * (1 - x)

def simple_nn(X, y, lr=0.05):
    np.random.seed(1)

    W1 = np.random.rand(X.shape[1], 2)
    W2 = np.random.rand(2, 1)

    for epoch in range(1000):
        hidden = sigmoid(np.dot(X, W1))
        output = sigmoid(np.dot(hidden, W2))

        error = y - output

        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden)

        W2 += hidden.T.dot(d_output) * lr
        W1 += X.T.dot(d_hidden) * lr

    return output


# -------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------

# AND Gate
X_and = np.array([
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
])
y_and = np.array([0,0,0,1])

weights = np.array([10, 0.2, -0.75])
lr = 0.05

# Train AND
final_w, epochs, errors = train_perceptron(X_and, y_and, weights, lr, step)

print("AND Gate Weights:", final_w)
print("Epochs:", len(epochs))

# Plot
plt.plot(epochs, errors)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("AND Gate Learning")
plt.show()


# Activation Comparison
activations = {
    "Step": step,
    "Bipolar": bipolar_step,
    "Sigmoid": sigmoid,
    "ReLU": relu
}

act_results = compare_activation(X_and, y_and, weights, lr, activations)
print("Activation Comparison:", act_results)


# Learning Rate
lr_vals, iter_vals = learning_rate_test(X_and, y_and, weights, step)

plt.plot(lr_vals, iter_vals)
plt.xlabel("Learning Rate")
plt.ylabel("Iterations")
plt.title("Learning Rate vs Iterations")
plt.show()


# XOR Gate (Fails in Perceptron)
X_xor = X_and.copy()
y_xor = np.array([0,1,1,0])

_, xor_epochs, _ = train_perceptron(X_xor, y_xor, weights, lr, step)
print("XOR Epochs (Perceptron):", len(xor_epochs))


# Customer Data
cust_data = np.array([
    [20,6,2,1],
    [16,3,6,1],
    [27,6,2,1],
    [19,1,2,0],
    [24,4,2,1],
    [22,1,5,0],
    [15,4,2,1],
    [18,4,2,1],
    [21,1,4,0],
    [16,2,4,0]
])

X_cust = cust_data[:, :3]
y_cust = cust_data[:, 3]
X_cust = np.insert(X_cust, 0, 1, axis=1)

weights_random = np.random.rand(4)

w_cust, _, _ = train_perceptron(X_cust, y_cust, weights_random, 0.01, sigmoid)
print("Customer Model Weights:", w_cust)


# Pseudo-Inverse
pi_weights = pseudo_inverse_solution(X_cust, y_cust)
print("Pseudo-Inverse Weights:", pi_weights)


# Neural Network (AND)
X_nn = np.array([[0,0],[0,1],[1,0],[1,1]])
y_nn = np.array([[0],[0],[0],[1]])

nn_output = simple_nn(X_nn, y_nn)
print("NN AND Output:\n", nn_output)


# Neural Network XOR
y_xor_nn = np.array([[0],[1],[1],[0]])
print("NN XOR Output:\n", simple_nn(X_nn, y_xor_nn))


# MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
mlp.fit(X_nn, y_nn.ravel())

print("MLP AND Prediction:", mlp.predict(X_nn))


# Example Dataset (replace with your TF-IDF)
# X_data, y_data assumed already available
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)

# mlp.fit(X_train, y_train)
# print("MLP Accuracy:", mlp.score(X_test, y_test))
