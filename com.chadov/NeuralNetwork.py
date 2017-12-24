from  numpy import *


class NeuralNetwork(object):
    def __init__(self):
        random.seed(1)
        # we model a single neuron, with 3 inputs and 1 output and assign random weight.
        self.weights = 2 * random.random((3, 1)) - 1

    def train(self, inputs, outputs, num):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * output * (1 - output))
            self.weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.weights))

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


network = NeuralNetwork()

# The training set
inputs = array([[1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[1, 1, 0]]).T

# Training the neural networ using the training set.
network.train(inputs, outputs, 30000)

# Ask the neural network the output
print(network.think(array([1, 0, 0])))
