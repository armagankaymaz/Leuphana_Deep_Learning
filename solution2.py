#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets
import torchvision.datasets

class ReLU:
    def evaluate(self, z):
        return z.clip(min=0)
    def gradient(self, z):
        tmp = np.zeros_like(z)
        tmp[z>0] = 1.0
        return tmp

class sigmoid:
    @staticmethod
    def evaluate(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def gradient(z):
        tmp = self.evaluate(z)
        return tmp*(1-tmp)
    
"""
# The NeuralNetwork class is supposed to work as follows.
# We can define a network, i.e., with 2 input, 3 hidden, and 2 output neurons as
net = NeuralNetwork([2,3,2])
# A forward pass (prediction) can be computed using
net.forward(x)
# and the gradients of all parameters can be obtained via
net.backprop(x, y)
# Finally, the network can be trained by calling
net.train(x, y)
# Other activation functions can be provided when creating the object:
net = NeuralNetwork([2,3,2], activations=[ReLU(), ReLU(), sigmoid()])
"""

#
# Task 5 (as a program)
#

# First layer
x0 = np.array([1.0, 2.0])
W1 = np.array([
    [0.2, -0.3],
    [0.1, -1.2],
    [0.4, 0.3],
])
b1 = np.array([0.3, -0.1, 1.2])
# W1 is (3, 2) and x0 is (2,)
s1 = W1 @ x0 + b1

# Second layer
x1 = sigmoid.evaluate(s1)
W2 = np.array([
    [0.4, 0.2, 0.2],
    [0.1, 0.3, 0.5],
])
b2 = np.array([-0.6, 0.5])
s2 = W2 @ x1 + b2
x2 = sigmoid.evaluate(s2)


# Task 6
# Part 1 
# (i)


# Task 6
# Part 1 
# (ii)

class NeuralNetwork:
    def __init__(self, neurons_per_layer, activations=None):
        self.neurons_per_layer = np.array(neurons_per_layer)
        self.weight = []
        self.bias = []
        self.activation = []
        self._s = []
        self._x = []
        for d_in, d_out in zip(neurons_per_layer[:-1], neurons_per_layer[1:]):
            # Alternative intialization:
            # - transform the variance of a standard Gaussian by rescaling it
            # self.weight.append(np.random.randn(d_out, d_in)*np.sqrt(2/(d_out+d_in)))
            self.weight.append(np.random.normal(
                loc=0.0,
                scale=np.sqrt(2.0 / (d_in + d_out)),
                size=(d_out, d_in),
            ))
            self.bias.append(np.zeros((d_out, 1)))
        for d in neurons_per_layer[:-1]:
            self._s.append(np.zeros((d, 1)))
        for d in neurons_per_layer:
            self._x.append(np.zeros((d, 1)))
        if activations==None:
            for _ in range(len(neurons_per_layer)-1):
                self.activation.append(sigmoid())
        else:
            self.activation = activations

    def forward(self, x):
        self._x[0] = x
        for l in range(len(self.neurons_per_layer)-1):
            self._s[l] = self.weight[l] @ self._x[l] + self.bias[l]
            self._x[l+1] = self.activation[l].evaluate(self._s[l])
        return self._x[-1]

    def loss(self, y_prediction, y):
        raise NotImplementedError

    def backprop(self, x, y):
        raise NotImplementedError

    def train(self, x, y, eta=.1, iterations=10):
        raise NotImplementedError



# you can also overwrite the parameters yourself
# to confirm that your implementation is correct,
# i.e., that it matches what you computed by hand
net = NeuralNetwork([2,3,2])
# set the weights and biases
net.weight[0] = np.array([[.2, -.3], [.1, -1.2], [.4, .3]])
net.bias[0] = np.array([[.3], [-.1], [1.2]])
net.weight[1] = np.array([[.4, .2, .2], [.1, .3, .5]])
net.bias[1] = np.array([[-.6], [.5]])
# set the input
x0 = np.array([[1.], [2.]])
# compute the forward pass
net.forward(x0)
# compare the intermediate states
net._s[0]
net._x[1]
net._s[1]
net._x[2]


#
# Task 6 Part 2 (i, ii)
#
# This impl. uses "arbitrary" rules to transform the derivatives to obtain the gradients
class NeuralNetworkOld:
    def __init__(self, neurons_per_layer, activations=None):
        self.neurons_per_layer = np.array(neurons_per_layer)
        self.weight = []
        self.bias = []
        self.activation = []
        self._s = []
        self._x = []
        for d_in, d_out in zip(neurons_per_layer[:-1], neurons_per_layer[1:]):
            self.weight.append(np.random.randn(d_out, d_in)*np.sqrt(2/(d_out+d_in)))
            self.bias.append(np.zeros((d_out, 1)))
        for d in neurons_per_layer[:-1]:
            self._s.append(np.zeros((d,1)))
        for d in neurons_per_layer:
            self._x.append(np.zeros((d,1)))
        if activations==None:
            for _ in range(len(neurons_per_layer)-1):
                self.activation.append(sigmoid())
        else:
            self.activation = activations

    def forward(self, x):
        self._x[0] = x
        for l in range(len(self.neurons_per_layer)-1):
            self._s[l] = self.weight[l] @ self._x[l] + self.bias[l]
            self._x[l+1] = self.activation[l].evaluate(self._s[l])
        return self._x[-1]

    def loss(self, y_prediction, y):
        value = 0.5*(y_prediction-y)**2
        gradient = y_prediction-y
        return value, gradient

    def backprop(self, x, y):
        y_prediction = self.forward(x)
        loss, loss_gradient = self.loss(y_prediction, y)
        print(f"loss = {loss.sum()}")
        weight_gradient = []
        bias_gradient = []
        num_layers = len(self.bias)
        for _ in range(num_layers):
            weight_gradient.append(None)
            bias_gradient.append(None)
        ones = np.ones((y.shape[1], 1))
        tmp = loss_gradient * self.activation[-1].gradient(self._s[-1])
        weight_gradient[-1] = tmp @ self._x[-2].T
        bias_gradient[-1] = tmp @ ones
        for l in reversed(range(num_layers-1)):
            tmp = self.activation[l].gradient(self._s[l]) * (self.weight[l+1].T @ tmp)
            weight_gradient[l] = tmp @ self._x[l].T
            bias_gradient[l] = tmp @ ones
        return weight_gradient, bias_gradient, loss

    def train(self, x, y, eta=.01, iterations=100):
        num_layers = len(self.bias)
        losses = []
        for iteration in range(iterations):
            print(f"iteration {iteration}")
            weight_gradient, bias_gradient, loss = self.backprop(x, y)
            for l in range(num_layers):
                self.weight[l] = self.weight[l] - eta * weight_gradient[l]
                self.bias[l] = self.bias[l] - eta * bias_gradient[l]
            losses.append(loss.sum())
        return losses


# Here we follow consistent derivative rules, i.e., shape of grad is always
# (out_size, in_size)
# where out_size is the size of the function outputs (what it maps to) and
# in_size is the size of the inputs (inputs is considered to be what we derive with)

class sigmoid2:
    def evaluate(self, z):
        return 1/(1+np.exp(-z))
    def gradient(self, z):
        tmp = self.evaluate(z)
        return tmp*(1-tmp)


class NeuralNetworkNew:
    def __init__(self, neurons_per_layer, activations=None):
        self.neurons_per_layer = np.array(neurons_per_layer)
        self.weight = []
        self.bias = []
        self.activation = []
        self._s = []
        self._x = []
        for d_in, d_out in zip(neurons_per_layer[:-1], neurons_per_layer[1:]):
            self.weight.append(np.random.randn(d_out, d_in)*np.sqrt(2/(d_out+d_in)))
            self.bias.append(np.zeros((d_out, 1)))
        for d in neurons_per_layer[:-1]:
            self._s.append(np.zeros((d,1)))
        for d in neurons_per_layer:
            self._x.append(np.zeros((d,1)))
        if activations==None:
            for _ in range(len(neurons_per_layer)-1):
                self.activation.append(sigmoid2())
        else:
            self.activation = activations

    def forward(self, x):
        self._x[0] = x
        for l in range(len(self.neurons_per_layer)-1):
            self._s[l] = self.weight[l] @ self._x[l] + self.bias[l]
            self._x[l+1] = self.activation[l].evaluate(self._s[l])
        return self._x[-1]

    def loss(self, y_prediction, y):
        value = 0.5*(y_prediction-y)**2
        gradient = y_prediction-y
        return value, np.reshape(gradient, (1, y.shape[0]))
    
    def backprop(self, x, y):
        y_prediction = self.forward(x)
        loss, loss_gradient = self.loss(y_prediction, y)
        weight_gradient = []
        bias_gradient = []
        num_layers = len(self.bias)
        delta = loss_gradient
        for l in reversed(range(num_layers)):
            grad_activation = self.activation[l].gradient(self._s[l])
            delta = delta * grad_activation  # Ensure element-wise multiplication
            weight_gradient_l = delta @ self._x[l].T
            bias_gradient_l = delta.sum(axis=1, keepdims=True)  # Sum over the batch dimension for biases
            if l > 0:
                delta = self.weight[l].T @ delta
            weight_gradient.insert(0, weight_gradient_l)
            bias_gradient.insert(0, bias_gradient_l)
        return weight_gradient, bias_gradient, loss

    def train(self, x, y, eta=0.01, iterations=100):
        losses = []
        for iteration in range(iterations):
            print(f"iteration {iteration+1}")
            x = x.reshape(self.neurons_per_layer[0], -1)  # Ensure x is shaped correctly
            for i in range(x.shape[1]):  # iterate over each sample
                xi = x[:, i:i+1]  # reshape each sample to (n_features, 1)
                yi = y[:, i:i+1]  # ensure y is correctly shaped
                weight_gradient, bias_gradient, loss = self.backprop(xi, yi)
                for l in range(len(self.bias)):
                    self.weight[l] -= eta * weight_gradient[l]
                    self.bias[l] -= eta * bias_gradient[l]
            losses.append(np.sum(loss))
        return losses


# Task 6 Part 2 (iii)
#

# construct toy data set
X, y = sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=3)
# or
X, y = sklearn.datasets.make_moons(n_samples=100, shuffle=True, noise=0.15, random_state=0)

# show the data set
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.plot(X[y==1,0], X[y==1,1], 'ro', label="positive class")
ax.plot(X[y==0,0], X[y==0,1], 'bo', label="negative class")
ax.legend()
plt.show()


# define a neural network with 2 input, 4 hidden and 1 output neuron
net = NeuralNetworkNew([2,4,1], activations=[ReLU(), sigmoid2()])
# train
loss = net.train(X.T, y.reshape(1, -1), eta=0.01, iterations=100)


fig, ax = plt.subplots()
ax.plot(np.arange(1,len(loss)+1), loss, 'k-')
plt.show()


fig, ax = plt.subplots()
ax.set_aspect(1)
ax.plot(X[y==1,0], X[y==1,1], 'ro', label="positive class")
ax.plot(X[y==0,0], X[y==0,1], 'bo', label="negative class")
ax.legend()
x_min = X[:, 0].min()-1
x_max = X[:, 0].max()+1
y_min = X[:, 1].min()-1
y_max = X[:, 1].max()+1
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
XY = np.vstack((XX.ravel(), YY.ravel())).T
# compute forward pass on every grid point
Z = net.forward(XY.T).reshape(XX.shape)
# show prediction area
ax.pcolormesh(XX, YY, Z, cmap=plt.cm.coolwarm)
# show separating line
ax.contour(XX, YY, Z, colors=['k'], linestyles=['-'], levels=[.5])
plt.show()



# count the parameters
number_of_parameters = 0
for k in range(len(net.neurons_per_layer)-1):
    number_of_parameters += net.weight[k].shape[0]*net.weight[k].shape[1]
    number_of_parameters += net.bias[k].shape[0]




#
# Task 7
#

def f(x):
    return np.sin(x)

# number of functions
n = 10
# beginning the the interval
a = 0
# end the the interval
b = 6
# beginning the the interval
x = np.linspace(a, b, n)
# spacing
h = x[1]-x[0]
# biases
b_ = x-h
# function values
f_ = f(x)

# matrix used in the linear system of equations
A = np.zeros((n, n))
for i in range(n):
    A[i:, i] = np.arange(1, n-i+1)

# fit parameters lambda
lambda_ = np.linalg.solve(h*A, f_)


def approx(x):
    # For more information, look for "list comprehension"
    return [np.sum(lambda_ * (x_i - b_).clip(min=0))
            for x_i in x]

# plot the function and the approximation
dom = np.linspace(-1, 8, 100)
fig, ax = plt.subplots()
ax.plot(dom, f(dom), 'b-')
ax.plot(dom, approx(dom), 'r-')
plt.show()
