from readfile import getImageInfo
from operations import sigmoid,softmax_crossentropy,backpro
import numpy as np
import time
import neuralnetwork
import matplotlib.pyplot as plt
train_images, train_labels = getImageInfo("train.mat")
test_images, test_labels = getImageInfo("test.mat")
validation_images, validation_labels = getImageInfo("validation.mat")

weights = np.random.random([5, 768])


print(validation_images.shape)
NN = neuralnetwork.Neural_Network()
start = time.time()
for k in range(1000):
    count =0
    for i in range(1): # trains the NN 1,000 times
      image = np.reshape(validation_images[i],(len(validation_images[i]),1)).T
      label = np.reshape(validation_labels[i],(len(validation_labels[i]),1)).T
      #print ("Input: \n" + str(image) )

      print ("Actual Output: \n" + str(label) )
      print ("Predicted Output: \n" + str(NN.forward(image)) )

      print ("Loss: \n" + str(np.mean(np.square(label- NN.forward(image))))) # mean sum squared loss
      print ("\n")
      result = np.equal(np.argmax(label), np.argmax(NN.forward(image)))
      if result == True:
          count+=1
      NN.train(image,label)
      if (np.mean(np.square(label- NN.forward(image)))) <0.008:
          print("program time ",time.time()-start)
          break
    print("??",count)
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