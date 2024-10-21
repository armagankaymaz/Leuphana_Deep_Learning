import numpy as np
import torch as th
#
# Task 1
#
x = th.full((13,13), 4, dtype=th.int)
x[:,[1,6,11]] = 5
x[[1,6,11],:] = 5
x[3:5,3:5] = 6
x[8:10,3:5] = 6
x[3:5,8:10] = 6
x[8:10,8:10] = 6
x

# it works similarly with numpy
x = np.ones((13,13), dtype=int)*4
x[:,[1,6,11]] = 5
x[[1,6,11],:] = 5
x[3:5,3:5] = 6
x[8:10,3:5] = 6
x[3:5,8:10] = 6
x[8:10,8:10] = 6




#
# Task 2
#
x = th.randn((3,1))
x
def sigmoid(x):
    return 1/(1+th.exp(-x))
def softmax(x):
    return th.exp(x)/th.exp(x).sum()

sigmoid(x)
softmax(x)


x = th.tensor([-3.,2,3,7])
sigmoid(x)
np.round(softmax(x),2)



#
# Task 3
#
import matplotlib.pylab as plt
import sklearn.datasets
import torchvision.datasets
from sklearn.svm import LinearSVC

###############################################################################

# generate toy data
X,y = sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=3) # random_state=42
# you can change or remove the random_state!

# change y=0 to y=-1
y[y==0] = -1

# subtract the mean of X from every data point => X has a mean of 0
X = X - X.mean(0)

"""
# tighter
X[y==+1] -= 0.2*X[y==+1].mean(0)
X[y==-1] -= 0.2*X[y==-1].mean(0)
# wider
X[y==+1] += 1*X[y==+1].mean(0)
X[y==-1] += 1*X[y==-1].mean(0)
"""

# plot toy data
fig, ax = plt.subplots()



# compute R
norms = np.linalg.norm(X, axis=1)
R = np.max(norms)

# compute gamma
svm = LinearSVC(C=1000, loss="hinge", tol=1e-5, random_state=0)
svm.fit(X, y)
margin = 1 / np.sqrt(np.sum(svm.coef_ ** 2))
gamma = margin

# get the separating hyperplane
w = svm.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (svm.intercept_[0]) / w[1]
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin


# plot toy data
fig, ax = plt.subplots()
ax.set_aspect(1)
#       first dim., second dim., dots,label
ax.plot(X[y==+1,0], X[y==+1,1], 'ro', label="positive class")
ax.plot(X[y==-1,0], X[y==-1,1], 'bo', label="negative class")
ax.plot(xx, yy, 'k-')
ax.plot(xx, yy_down, 'k--')
ax.plot(xx, yy_up, 'k--')
ax.plot([0,X[np.argmax(norms),0]], [0,X[np.argmax(norms),1]], 'k-')
circle = plt.Circle((0, 0), R, color='k', fill=False)
ax.add_artist(circle)
ax.legend()
plt.show()

###############################################################################

# add 1 to each data point
ones = np.ones((X.shape[0], 1))
X = np.hstack((X, ones))

# initialize the weight vector w
w = np.array([0.0, 1.0, 0.0])

def f(X, w):
    return np.dot(X, w)

def classifier(X, w):
    return np.sign(f(X, w))

# domain
dom = np.linspace(X[:,0].min()-2, X[:,0].max()+2, 10)

def discriminant(x1, w):
# f(x) = w*x+b
#      = w0*x0 * w1*x1 + w2*1
# f(x) != 0
# solve for x1, i.e. x1 = ...
# x2 = -w0/w1 x0 - b/w1
    x2 = -x1*w[0]/w[1] - w[2]/w[1]
    return x2

# show the decision hyperplane for the current w
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlim(X[:,0].min()-1,X[:,0].max()+1)
ax.set_ylim(X[:,1].min()-1,X[:,1].max()+1)
ax.plot(X[y==+1,0], X[y==+1,1], 'ro', label="positive class")
ax.plot(X[y==-1,0], X[y==-1,1], 'bo', label="negative class")
ax.plot(dom, discriminant(dom, w), 'k-' ) # k is short for black
ax.legend()
plt.show()


# implementation of the perceptron
def perceptron(X, y, w):
    converged = False
    n = X.shape[0]
    total_updates = 0
    updates = 0
    while not converged:
        for i in range(n):
            if y[i]*f(X[i], w) <= 0:
                w = w + X[i]*y[i]
                updates += 1
        if updates == 0:
            converged = True
        total_updates += updates
        updates = 0

    return w, total_updates

w, total_updates = perceptron(X, y, w)


# show the decision hyperplane for the learned w
fig, ax = plt.subplots()
ax.set_aspect(1)
ax.set_xlim(X[:,1].min()-1,X[:,1].max()+1)
ax.set_ylim(X[:,2].min()-1,X[:,2].max()+1)
ax.plot(X[y==+1,1], X[y==+1,2], 'ro', label="positive class")
ax.plot(X[y==-1,1], X[y==-1,2], 'bo', label="negative class")
ax.plot(dom, discriminant(dom, w), 'k-') # k is short for black
ax.legend()
plt.show()

###############################################################################

# run the perceptron several times with random w initializations
fig, ax = plt.subplots(figsize=(15,10))
ax.set_aspect(1)
ax.set_xlim(X[:,1].min()-1,X[:,1].max()+1)
ax.set_ylim(X[:,2].min()-1,X[:,2].max()+1)
ax.plot(X[y==+1,1], X[y==+1,2], 'ro', label="positive class")
ax.plot(X[y==-1,1], X[y==-1,2], 'bo', label="negative class")
res = []
for j in range(250):
    w = np.random.uniform(-20,20,3)
    w, total_updates = perceptron(X, y, w)
    res.append(total_updates)
    ax.plot(dom, discriminant(dom, w), 'k-', alpha=0.1) # k is short for black
ax.legend(loc=2)
plt.show()


# set the bound
bound = 4*R**2/gamma**2
# does the bound hold for all runs on this data set?
print(np.all(res <= bound))

fig, ax = plt.subplots()
ax.hist(res, density=True)
plt.show()

###############################################################################

# now do the same experiment but with a wider and tighter corridor

"""
# tighter
X[y==+1] -= 0.2*X[y==+1].mean(0)
X[y==-1] -= 0.2*X[y==-1].mean(0)
# wider
X[y==+1] += 1*X[y==+1].mean(0)
X[y==-1] += 1*X[y==-1].mean(0)
"""

###############################################################################

mnist = torchvision.datasets.MNIST('./data', download=True)
X = mnist.data
y = mnist.targets

fig, ax = plt.subplots()
ax.imshow(X[0], cmap='gray')
plt.show()

subset_indices = np.hstack((np.where(y==6)[0],
                            np.where(y==9)[0]))


# fix the data set
X = X[subset_indices].numpy()
y = y[subset_indices].numpy()
y[y==9] = 1
y[y==6] = -1
X = X.reshape(X.shape[0], -1)

"""
# subtract the mean of X from every data point => X has a mean of 0
X = X - X.mean(0)

# compute R
norms = np.linalg.norm(X, axis=1)
R = np.max(norms)

# compute gamma
svm = LinearSVC(C=1000, loss="hinge", tol=1e-5, random_state=0)
svm.fit(X, y)
margin = 1 / np.sqrt(np.sum(svm.coef_ ** 2))
gamma = margin

# set the bound
bound = 4*R**2/gamma**2
"""

# add 1 to each data point
X = np.hstack((np.ones(X.shape[0]).reshape(-1,1), X))

# initialize the weight vector w
w = np.random.randn(X.shape[1])

# run the perceptron
w, total_updates = perceptron(X, y, w)

# check the classification error on train data
np.all(classifier(X, w) == y)
np.mean(classifier(X, w) == y)

# load test data
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
X_test = mnist_test.data
y_test = mnist_test.targets
subset_indices_test = np.hstack((np.where(y_test==6)[0],
                                 np.where(y_test==9)[0]))
X_test = X_test[subset_indices_test].numpy()
y_test = y_test[subset_indices_test].numpy()
y_test[y_test==6] = -1
y_test[y_test==9] = +1
X_test = X_test.reshape(X_test.shape[0], -1)
X_test = np.hstack((np.ones(X_test.shape[0]).reshape(-1,1), X_test))


# check the classification error on test data
np.all(classifier(X_test, w) == y_test)
np.mean(classifier(X_test, w) == y_test)


missclassified_indices = np.where(classifier(X_test, w) != y_test)[0]
for idx in missclassified_indices:
    fig, ax = plt.subplots()
    ax.imshow(X_test[idx,1:].reshape(28,28), cmap='gray')
    plt.show()


