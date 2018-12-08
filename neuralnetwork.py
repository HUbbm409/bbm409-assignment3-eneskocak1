import numpy as np



class Neural_Network(object):
    def __init__(self,layersize,nodesize):
        #parameters
        self.inputSize = 768
        self.outputSize = 5

        self.hiddenSize = nodesize

        #weights
        self.W=list()
        self.B=list()
        if layersize > 0 :
            for i in range(layersize+1):
                if i ==0:

                    self.W.append(np.random.randn(self.inputSize, self.hiddenSize))# (3x2) weight matrix from input to hidden layer
                    self.B.append(np.random.randn(1,self.hiddenSize))
                elif i == layersize:
                    self.W.append(np.random.randn(self.hiddenSize, self.outputSize)) # (3x1) weight matrix from hidden to output layer
                    self.B.append(np.random.randn(1,self.outputSize))
                else:
                    self.W.append(np.random.randn(self.hiddenSize, self.hiddenSize))
                    self.B.append(np.random.randn(1,self.hiddenSize))
        else:
            self.W.append(np.random.randn(self.inputSize, self.outputSize))
            self.B.append(np.random.randn(1,self.outputSize))

    def forward(self, X):
        #forward propagation through our network
        self.inputs=list()
        self.inputs.append(X)
        self.z = np.dot(X, self.W[0]) # dot product of X (input) and first set of 3x2 weights
        self.z += self.B[0]
        self.z = self.sigmoid(self.z)  # activation function
        for i in range(1,len(self.W)):
            self.inputs.append(self.z)
            self.z = np.dot(self.z, self.W[i]) # dot product of hidden layer (z2) and second set of 3x1 weights
            self.z+= self.B[i]
            self.z = self.sigmoid(self.z)  # activation function
        return self.z

    def sigmoid(self, x):
        # activation funxtion
        return 1/(1+np.exp(-x))

    def sigmoidPrime(self, x):
        # derivative of sigmoid
        return x * (1 - x)

    def backward(self, y,o, error,delta_error,rate):
        # backward propgate through the network
        self.delta_error = delta_error  # error in output
        self.o_delta =rate* error*self.delta_error * self.sigmoidPrime(o)  # applying derivative of sigmoid to error
        for i in reversed(range(len(self.W))):
            #print("\n\n")
            #print("delta",self.o_delta)
            #print("weight",self.W[i])
            self.o_error = self.o_delta.dot(self.W[i].T)  # z2 error: how much our hidden layer weights contributed to output error
            #print("input",self.inputs[i].T.dot(self.o_delta))

            self.B[i] += self.o_delta

            self.W[i] += self.inputs[i].T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights
            self.o_delta = self.o_error * self.sigmoidPrime(self.inputs[i])  # applying derivative of sigmoid to z2 error


    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        """ sum of them 1 """
        softmax = np.exp(x) / np.sum(np.exp(x))

        return softmax

    def train(self, X, y,rate):
        output = self.forward(X)
        error, out = self.cross_entropy(output, y)
        delta_error = self.delta_cross_entropy(output, y)


        self.backward( y, output,error,delta_error,rate)


        loss = np.sum(error)/error.shape[0]
        return loss,out

    def cross_entropy(self,X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """

        p = self.softmax(X)
        #E = – ∑ ci . log(pi) + (1 – ci ). log(1 – pi)
        #print("softmax ",p)
        log_likelihood =-((y*np.log10(p) ) + ((1-y) * np.log10(1-p)))

        return log_likelihood,p.argmax(axis=1)

    def delta_cross_entropy(self,X, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector.
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """

        grad = self.softmax(X)
        result = y-grad

        return result