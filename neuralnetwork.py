import numpy as np



class Neural_Network(object):
    def __init__(self,layersize,nodesize,activation,shape1,shape2):
        #parameters
        self.inputSize = shape1
        self.outputSize = shape2
        self.batch= 0
        self.batcherror= np.zeros((1,shape2))
        self.hiddenSize = nodesize
        self.layersize =layersize

        if activation == "sigmoid":
            self.activationfunc=self.sigmoid
            self.activationDerivative = self.derivative_sigmoid
        elif activation == "relu":
            self.activationfunc = self.ReLU
            self.activationDerivative = self.derivative_Relu

        self.W=list()
        self.B=list()

        for i in range(self.layersize+1):
            if i ==0:
                if layersize !=0:
                    self.W.append(2*np.random.random([self.inputSize, self.hiddenSize])-1)# (3x2) weight matrix from input to hidden layer
                    self.B.append(np.ones((1,self.hiddenSize)))

                else:
                    self.W.append(2*np.random.random([self.inputSize,
                                                  self.outputSize])-1)  # (3x2) weight matrix from input to hidden layer
                    self.B.append(np.ones((1,self.outputSize)))

            elif i == layersize:
                self.W.append(2*np.random.random([self.hiddenSize, self.outputSize])-1) # (3x1) weight matrix from hidden to output layer
                self.B.append(np.ones((1,self.outputSize)))
            else:
                self.W.append(2*np.random.random([self.hiddenSize, self.hiddenSize])-1)
                self.B.append(np.ones((1,self.hiddenSize)))

    def forward(self, X):
        #forward propagation through our network

        self.outsa=list()
        self.z = X
        self.outsa.append(X)
        for i in range(len(self.W)):

            self.z = self.z.dot(self.W[i])
            self.z+= self.B[i]
            self.z = self.activationfunc(self.z)
            self.outsa.append(self.z)
        outsoft = self.softmax(self.z)

        return outsoft

    def sigmoid(self, x):
        # activation funxtion
        e= 1 / (1 + np.exp(-x))
        return e

    def derivative_sigmoid(self, x):
        # derivative of sigmoid
        return x * (1 - x)

    def backward(self,output,alpha):

        self.delta = output
        self.delta *= alpha
        self.delta *= self.activationDerivative(self.outsa[self.layersize+1])
        for i in reversed(range(len(self.W))):

            self.B[i] += self.delta
            self.o_error = self.outsa[i].T.dot(self.delta)
            self.delta = self.delta.dot(self.W[i].T) * self.activationDerivative(self.outsa[i])
            self.W[i] += self.o_error

    def softmax(self,x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def ReLU(self,x):

        return x * (x > 0)

    def derivative_Relu(self,x):

        return 1*(x>0)


    def train(self, X, y,rate,batchsize):
        output = self.forward(X)
        loss=self.cross_entropy(output,y)
        delta=self.delta_cross_entropy(output,y)

        self.batch +=1
        self.batcherror += delta

        if batchsize == self.batch :
            self.batcherror /= batchsize
            self.backward(self.batcherror,rate)
            self.batch=0
            self.batcherror[:]=0


        return output,loss


    def cross_entropy(self,out,y):

        #E = – ∑ ci . log(pi) + (1 – ci ). log(1 – pi)
        log_likelihood =-((y*np.log10(out) ) + ((1-y) * np.log10(1-out)))

        return log_likelihood.sum()

    def delta_cross_entropy(self,soft, y):
        result = y-soft

        return result