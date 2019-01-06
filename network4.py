"""network2.py
~~~~~~~~~~~~~~

An improved version of network2.py, implementing the layers.
features.

"""

# Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


# Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y.reshape) * Sigmoid(z).prime()


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return a - y.reshape(a.shape)


class ReLU(object):
    def __init__(self, z):
        self.z = z

    def fn(self):
        return np.maximum(0, self.z)

    def prime(self):
        return np.sign(self.fn())


class Sigmoid(object):
    def __init__(self, z):
        self.z = z

    def fn(self):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-self.z))

    def prime(self):
        """Derivative of the sigmoid function."""
        return self.fn() * (1 - self.fn())


class Variable(np.ndarray):
    """This class is inherit from np.ndarray to save value and gradient at the
    same time. The class of data must be np.ndarray.

    """

    def __new__(cls, array):
        return super(Variable, cls).__new__(cls, array.shape)

    def __init__(self, array):
        super(Variable, self).__init__()
        self[:] = array.copy()
        self.grad = np.zeros(self.shape)

    def zero_grad(self):
        self.grad = np.zeros(self.shape)


class Layer(object):
    """
    super class for all layers.
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ToFC(Layer):
    def __init__(self):
        self.shape = None
        self.params = [Variable(np.ndarray((1,)))]

    def forward(self, inpt):
        self.shape = inpt.shape
        return inpt.reshape((-1, 1))

    def backward(self, delta):
        return delta.reshape(self.shape)


class MaxPoolingLayer(Layer):
    def __init__(self):
        self.mask = None
        self.params = [Variable(np.ndarray((1,)))]

    def forward(self, inpt, kernel_size=2):
        inpt = inpt.transpose((2, 0, 1))
        shape = inpt.shape
        self.mask = np.zeros(shape)
        out = np.zeros((shape[0], shape[1] // 2, shape[2] // 2))
        for i in range(shape[0]):
            for j in range(0, shape[1], kernel_size):
                for k in range(0, shape[2], kernel_size):
                    window = inpt[i][j:j + kernel_size, k:k + kernel_size].reshape(-1)
                    index = np.argmax(window)
                    self.mask[i][j + index // kernel_size][k + index % kernel_size] = 1
                    out[i][j // kernel_size][k // kernel_size] = np.max(window)
        return out.transpose((1, 2, 0))

    def backward(self, delta):
        delta = delta.transpose((2, 0, 1))
        shape = delta.shape
        out = np.zeros((shape[0], shape[1] * 2, shape[2] * 2))
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    out[i][j * 2][k * 2] = out[i][j * 2 + 1][k * 2] = out[i][j * 2][k * 2 + 1] = out[i][j * 2 + 1][
                        k * 2 + 1] = delta[i][j][k]
        return (out * self.mask).transpose((1, 2, 0))


class CNNLayer(Layer):
    """
    A convolutional neural network layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, activate=Sigmoid, stride=1, padding=0, to_fc=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding: 
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel = Variable(np.random.randn(out_channels, kernel_size, kernel_size, in_channels))
        self.activate = activate
        self.stride = stride
        self.padding = padding
        self.params = [self.kernel]
        self.to_fc = to_fc
        # the attributes below is initialized  in the other methods
        self.inpt = None
        self.z = None
        self.y = None

    def forward(self, inpt):
        # shape of input: [width, height, in_channel]
        self.inpt=inpt.copy()

        inpt = np.array([np.pad(x, self.padding, 'constant') for x in inpt.transpose((2, 0, 1))]).transpose((1, 2, 0))
        shape = inpt.shape
        try:
            assert shape[2] == self.in_channels
        except AssertionError:
            raise Exception("Shape[2] of input data is {}, but in_channels is {}.".format(shape[0], self.in_channels))
        # note that the shape of z is [channel, weight, height]
        self.z = np.zeros((self.out_channels, (shape[0] - self.kernel_size) // self.stride + 1, (
                shape[1] - self.kernel_size) // self.stride + 1))
        for i in range(self.out_channels):
            self.z[i] = self.conv(inpt, self.kernel[i], self.stride)
        self.z = self.z.transpose((1, 2, 0))
        self.y = self.activate(self.z).fn()
        return self.y

    def backward(self, delta):
        delta_z = delta * self.activate(self.z).prime()
        kernel_grad = np.zeros((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        for i, delta_out in enumerate(delta_z.transpose((2, 0, 1))):
            for j, delta_in in enumerate(self.inpt.transpose((2, 0, 1))):
                kernel_grad[i][j] = self.conv(delta_in, delta_out, self.stride)
        self.kernel.grad += kernel_grad.transpose((0, 2, 3, 1))
        in_grad = np.zeros(self.inpt.shape).transpose((2, 0, 1))
        delta_z = np.array([np.pad(x, self.padding, 'constant') for x in delta_z.transpose((2, 0, 1))]).transpose((1, 2, 0))
        for i, each_kernel in enumerate(self.kernel.transpose((3, 1, 2, 0))):
            in_grad[i] = self.conv(delta_z, each_kernel, self.stride)
        return in_grad.transpose((1, 2, 0))

    @staticmethod
    def pad(data, padding):
        return np.pad(data, padding, 'constant')

    @staticmethod
    def conv(inpt, kernel, stride):
        shape = inpt.shape
        kernel_size = kernel.shape
        out = np.zeros(((shape[0] - kernel_size[0]) // stride + 1, (shape[1] - kernel_size[1]) // stride + 1))
        for i in range((shape[0] - kernel_size[0]) // stride + 1):
            for j in range((shape[1] - kernel_size[1]) // stride + 1):
                # 在此处，为了简化卷积层计算代码，输入输出的数据形式必须为 [width, height, channel]
                out[i][j] = np.sum(inpt[i:i + kernel_size[0], j:j + kernel_size[1]] * kernel)
        return out


class FullyConnectedLayer(Layer):
    """A fully connect layer.
    TODO: separate activation from this class just as what pytorch have done.

    """

    def __init__(self, n_in, n_out, activate=Sigmoid):
        """
        :param n_in:
        :param n_out:
        :param activate:

        """
        self.n_in = n_in
        self.n_out = n_out
        # self.w = Variable(np.random.randn(self.n_out, self.n_in) / np.sqrt(self.n_in))
        self.w = Variable(np.random.randn(self.n_out, self.n_in))
        self.b = Variable(np.random.randn(self.n_out, 1))
        self.params = [self.w, self.b]
        self.y = np.zeros((n_out, 1))
        self.z = np.zeros((n_out, 1))
        self.x = np.zeros((n_in, 1))
        self.activate = activate

    def backward(self, delta):
        """
        :param delta: derivative of y, where y=f(z)=f(wx+b) and f is
        activation function.
        :return:derivative of x, which is the output of front layer.
        """
        # derivative of z
        delta = delta * self.activate(self.z).prime()
        self.w.grad += np.dot(delta, self.x.transpose())
        self.b.grad += delta.copy()
        return np.dot(self.w.transpose(), delta)

    def forward(self, inpt):
        """
        :param inpt: input data, I have not implement mini_batch yet.
        :return:
        """
        self.x = inpt
        self.z = np.dot(self.w, inpt) + self.b
        return self.activate(self.z).fn()


def mse_loss(output, y):
    """
    :param output: the output of the net.
    :param y: the label of input.
    :return: \partial L/ \partial output
    """
    return output - y


class TestNetwork(object):

    def __init__(self, layers, batch_size, loss_fn=mse_loss):
        self.layers = layers
        self.layer_len = len(layers)
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.params = [param for layer in self.layers for param in layer.params]

    def forward(self, batch):
        """
        :param x: input data, size is (mini_batch, size, 1).
        :return: probability of which class this image belongs to.
        """
        output = np.zeros((self.batch_size, self.layers[-1].y.shape[0], 1))
        total_loss = 0
        for i, (data, label) in enumerate(batch):
            for layer in self.layers:
                data = layer(data)
            output[i] = data
            # backward
            delta = self.loss_fn(data, label)
            for i in range(self.layer_len - 1, -1, -1):
                delta = self.layers[i].backward(delta)
            total_loss += delta

        return output, total_loss

    def backward(self):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class SGD(object):
    def __init__(self, params, eta, lmbda=0.0):
        self.params = params
        self.eta = eta
        self.lmbda = lmbda

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    def step(self):
        for i in range(len(self.params)):
            self.params[i] = self.params[i] - self.eta * self.lmbda * self.params[i] - self.eta * self.params[i].grad


class Loss(object):
    """In pytorch, loss is calculated by compute graph, which means all the
    information can be find in parameter output which is passed to update
    function. But in my version, output is only numbers, so this class need
    the information of the network.

    For simplicity of my code, this class is useless.
    """

    def __init__(self, network):
        self.network = network

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        raise NotImplementedError

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        raise NotImplementedError

    def backward(self, output, y):
        assert len(output) == len(y)
        # delta = self.delta(self.network.layers[-1].z, output, y)
        self.network.backward(y - output)

    def __call__(self, output, y):
        return self.backward(output, y)


class MSELoss(Loss):
    """Mean square loss. We don't need __init__ method because python will
    automatically call __init__ in superclass if no __init__ method is implied
    in subclss.

    """

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer. We assume that the
        last layer will always use sigmoid."""
        return (a - y) * Sigmoid(z).prime()


# Loading a Network
# def load(filename):
#     """Load a neural network from the file ``filename``.  Returns an
#     instance of Network.
#
#     """
#     f = open(filename, "r")
#     data = json.load(f)
#     f.close()
#     cost = getattr(sys.modules[__name__], data["cost"])
#     net = Network(data["sizes"], cost=cost)
#     net.weights = [np.array(w) for w in data["weights"]]
#     net.biases = [np.array(b) for b in data["biases"]]
#     return net

