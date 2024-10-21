#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pylab as plt
import sklearn.datasets
import torchvision.datasets


#
# Task 8
#


# load train data
mnist = torchvision.datasets.MNIST('./data', download=True)
X = mnist.data
y = mnist.targets
subset_indices = np.hstack((np.where(y==6)[0], np.where(y==9)[0]))
X = X[subset_indices].numpy()
y = y[subset_indices].numpy()
y[y==9] = 1
y[y==6] = 0
X = X.reshape(X.shape[0], -1)


# load test data
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
X_test = mnist_test.data
y_test = mnist_test.targets
subset_indices_test = np.hstack((np.where(y_test==6)[0], np.where(y_test==9)[0]))
X_test = X_test[subset_indices_test].numpy()
y_test = y_test[subset_indices_test].numpy()
y_test[y_test==9] = 1
y_test[y_test==6] = 0
X_test = X_test.reshape(X_test.shape[0], -1)

# Neural Network Class (same as previous exercise)
class sigmoid2:
    def evaluate(self, z):
        return 1/(1+np.exp(-z))
    def gradient(self, z):
        tmp = self.evaluate(z)
        return tmp*(1-tmp)


class NeuralNetwork:
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

# define the network and train
net = NeuralNetwork([X.shape[1],256,32,1])
loss = net.train(X.T, y.reshape(1,-1), eta=0.00025, iterations=100)

# show the loss during training
fig, ax = plt.subplots()
ax.plot(np.arange(1,len(loss)+1), loss, 'k-')
plt.show()

# (i)
# compute the train accuracy
pred = net.forward(X.T) > 0.5
np.mean(y == pred)

# compute the test accuracy
pred = net.forward(X_test.T) > 0.5
np.mean(y_test == pred)


# (ii)
# select a '9'
i = 5923
fig, ax = plt.subplots()
ax.imshow(X[i].reshape(28,28), cmap='gray')
plt.show()

from PIL import Image

def rotate(x, angle):
    img = Image.fromarray(x.reshape(28,28))
    rotated_img = img.rotate(angle)
    return np.array(rotated_img).reshape(-1,1)


fig, ax = plt.subplots(1,6)
for j, angle in enumerate([0,45,90,180,270,360]):
    ax[j].imshow(rotate(X[i], angle).reshape(28,28), cmap='gray')
    ax[j].set_axis_off()
    ax[j].set_title(f"{angle} deg.")
plt.show()
# fig.savefig('/tmp/rotation.pdf', bbox_inches='tight')


# compute the prediction for a given angle
angles = np.linspace(0, 360, 100)
prob_9 = [net.forward(rotate(X[i], angle)).item() for angle in angles]


fig, ax = plt.subplots()
ax.plot(angles, prob_9, 'b-')
ax.set_xlabel('degree')
ax.set_ylabel('probability of being 9')
ax.axhline(0.5, ls='-', c='k')
plt.show()
#fig.savefig('/tmp/rotation.png', dpi=dpi, transparent=False, bbox_inches='tight')
fig.savefig('/tmp/rotation_probability.pdf', bbox_inches='tight')


#
# Task 9
#

class sigmoid:
    def evaluate(self, z):
        return 1/(1+np.exp(-z))
    def gradient(self, z):
        tmp = self.evaluate(z)
        return tmp*(1-tmp)

class softmax:
    def evaluate(self, z): # shape of z is (hidden, batchsize)
        return np.exp(z) / np.exp(z).sum(0)

class SoftmaxCrossEntropyLoss:
    def evaluate(self, p, y, eps=1e-12):
        return -np.sum(y*np.log(p+eps), axis=0)
    def gradient(self, p, y):
        return p-y


class NeuralNetwork:
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
        if isinstance(self.activation[-1], softmax):
            # last layer is a softmax layer
            print("Info: softmax layer detected. Using CrossEntropyLoss.")
        else:
            # use squared loss
            print("Info: Using SquaredLoss.")

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

        weight_gradient = []
        bias_gradient = []
        for _ in range(len(self.bias)):
            weight_gradient.append([])
            bias_gradient.append([])

        if isinstance(self.activation[-1], softmax):
            # last layer is a softmax layer
            f = SoftmaxCrossEntropyLoss()
            loss = f.evaluate(y_prediction, y)
            loss_gradient = f.gradient(y_prediction, y)
            tmp = loss_gradient
        else:
            # use squared loss
            loss, loss_gradient = self.loss(y_prediction, y)
            tmp = loss_gradient * self.activation[-1].gradient(self._s[-1])
            
        print(f"loss = {loss.sum()}")

        ones = np.ones((y.shape[1], 1))
        weight_gradient[-1] = tmp @ self._x[-2].T
        bias_gradient[-1] = tmp @ ones
        for k in reversed(range(len(self.neurons_per_layer)-2)):
            tmp = self.activation[k].gradient(self._s[k]) * (self.weight[k+1].T @ tmp)
            weight_gradient[k] = tmp @ self._x[k].T
            bias_gradient[k] = tmp @ ones
        return weight_gradient, bias_gradient, loss

    def train(self, x, y, eta=.01, iterations=100):
        losses = []
        for iteration in range(iterations):
            print(f"iteration {iteration}")
            weight_gradient, bias_gradient, loss = self.backprop(x, y)
            for k in range(len(self.neurons_per_layer)-1):
                self.weight[k] = self.weight[k] - eta * weight_gradient[k]
                self.bias[k] = self.bias[k] - eta * bias_gradient[k]
            losses.append(loss.sum())
        return losses




# load train data
mnist = torchvision.datasets.MNIST('./data', download=True)
X = mnist.data.numpy()
y = mnist.targets.numpy()
X = X.reshape(X.shape[0], -1)
n = X.shape[0]

# restrict data set size
n = 100
X = X[:n]
y = y[:n]

# one-hot encoding
tmp = np.zeros((n,10), dtype=np.int)
tmp[range(n),y] = 1
y = tmp

# define neural network
net = NeuralNetwork([X.shape[1], 512, 10], activations=[sigmoid(), softmax()])
# train
loss = net.train(X.T, y.T, eta=0.025, iterations=1000)

# show the loss during training
fig, ax = plt.subplots()
ax.plot(np.arange(1,len(loss)+1), loss, 'k-')
plt.show()

# predict
p = net.forward(X.T)
# compare prediction to true labels
np.all(p.argmax(0) == y.T.argmax(0))

np.mean(p.argmax(0) == y.T.argmax(0)) 


#
# Task 10
#

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 128

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')




def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


k = 100

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
    #for i, (inputs, labels) in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        #inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % k == (k-1):    # print every k mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / k))
            running_loss = 0.0
    # here things are done per epoch
    print('Epoch ended')

print('Finished Training')



images, labels = next(dataiter)
outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1)
predicted
labels

# show missclassified samples
idx = np.where(predicted.cpu().numpy() != labels.numpy())[0]
for i in idx:
    imshow(torchvision.utils.make_grid(images[i]))
