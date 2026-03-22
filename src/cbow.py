import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def init_weights(V, N=50):
    W1 = np.random.randn(N, V) * 0.01
    b1 = np.zeros((N, 1))
    W2 = np.random.randn(V, N) * 0.01
    b2 = np.zeros((V, 1))
    return W1, b1, W2, b2

def one_hot(word, word_to_ix, V):
    vec = np.zeros(V)
    vec[word_to_ix[word]] = 1
    return vec