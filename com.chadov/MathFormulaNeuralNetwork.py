from  numpy import *


# Teaching the computer to predict the output
# of a mathematical expression without "knowing"
# exact formula (a+b)*2

class NeuralNetwork(object):
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((2, 1)) - 1

    # Takes the inputs and corresponding
    # outputs and trains the network num times
    # During each iteration we calculate the output using
    # the "think" method, calculate the error and adjust the weights
    # using the formula: adjustment = 0.01 * error * input
    def train(self, inputs, outputs, num):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = 0.01 * dot(inputs.T, error)  # T is the transpose of the matrix
            # it transposes the matrix from horizontal to vertical
            self.weights += adjustment

    # Calculates the weighted sum using the formula:
    # weight1 * input1 + weight 2 * input2
    def think(self, inputs):
        return dot(inputs, self.weights)


neural_network = NeuralNetwork()

inputs = array([[2, 3], [1, 1], [5, 2], [12, 3]])
outputs = array([[10, 4, 14, 30]]).T
number_of_iterations = 10000

print("Train our network 10000 times\n")
neural_network.train(inputs, outputs, number_of_iterations)

print("Try to think with array[15,2]\n")
print(neural_network.think(array([15, 2])))
print("now our network work like it knows the formula (a + b)* 2 ")

print("What T does mean : we ' ll print outputs\n")
print(outputs)
print("and print it without T\n")
print(array([[10, 4, 14, 30]]))
