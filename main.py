from readfile import getImageInfo

import numpy as np
import time
import neuralnetwork
import matplotlib.pyplot as plt
train_images, train_labels = getImageInfo("train.mat")
test_images, test_labels = getImageInfo("test.mat")
validation_images, validation_labels = getImageInfo("validation.mat")

#asdasdaaasd
deneme = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
lab= np.array([[1,0],
              [0,1],
              [0,1],
              [0,1]])


def nn(epoch,filetype,labeltype,batch):
    NN = neuralnetwork.Neural_Network(0,10)
    start = time.time()
    for k in range(epoch):
        count =0
        cross_entropy_loss = 0
        for i in range(len(filetype)//batch): # trains the NN 1,000 times

            image = filetype[i*batch:((i+1)*batch)-1]
            label = labeltype[i*batch:((i+1)*batch)-1]

            loss,predcit = NN.train(image, label,0.01)

            """print ("Actual Output:",label.argmax(axis=1) )
            print ("Predicted Output:" ,predcit)"""

            if np.all(label.argmax(axis=1) == predcit)==True:
                count+=1
            cross_entropy_loss += loss


        """print("LOSS",cross_entropy_loss/len(lab))
        print("hit :",count)"""

    print(count)

nn(1000,train_images,train_labels,10)
"""
layer s=[]
for i in range(1,599):


    enes = validation_images[i]
    enes = np.reshape(enes, (len(enes),1))

    perceptrons = np.dot(weights,enes)
    print(perceptrons)
    perceptrons=sigmoid(perceptrons)



    res=softmax_crossentropy(perceptrons,validation_labels[i])


    layers.append(np.argmax(perceptrons))
    back = backpro(enes,res)
    print("weights",weights)
    print("back",back)
    weights += back
    input("a")
    print(perceptrons)







enes = neuralnetwork.NeuralNetwork(validation_images,validation_labels)
for i in range(134):
    enes.feedforward()
    enes.backprop()

print((enes.weights1/10000000).round())
for i in range(enes.weights1.shape[1]):
    ekones = (enes.weights1).T[i]

    ekones = np.reshape(ekones,(32,24))
    print(ekones)

    plt.imshow(ekones,cmap="gray")
    plt.show()"""