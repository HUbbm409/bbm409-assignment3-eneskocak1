import numpy as np



def sigmoid(x):

    value = 1/(1+np.exp(-x))

    return value


def error(labels,x):

    result = labels-sigmoid(x)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    """ sum of them 1 """
    loss = np.exp(x) / np.sum(np.exp(x))

    return loss

def sigmoid_derivative(x):
    result= x*(1-x)

    return result


def softmax_crossentropy(neurons, label):

    label=np.reshape(label,(len(label),1))
    print(label)
    print(softmax(neurons))
    return ((label-softmax(neurons))**2)/2

def backpro(perceptrons,x):
    first = 2*x
    print("cross :",first.shape)
    print("sigmoid",sigmoid_derivative(first).shape)
    second = sigmoid_derivative(first)

    last = np.dot(second, perceptrons.T)
    return last