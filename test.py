from readfile import getImageInfo
import pickle
import numpy as np
import neuralnetwork
import time

train_images, train_labels = getImageInfo("train.mat")
test_images, test_labels = getImageInfo("test.mat")
validation_images, validation_labels = getImageInfo("validation.mat")



def getPikle(l,n,active):
    if l != 0 :
        with open(active+"_"+str(l)+"layer_"+str(n)+"node"+".pkl", "rb") as inp:
            NN = pickle.load(inp)
    else:
        with open(active+"_singlelayer.pkl", "rb") as inp:
            NN = pickle.load(inp)
    return NN

def setPikle(NN,activation):
    if NN.layersize !=0:
        with open(activation+"_"+str(NN.layersize)+"layer_"+str(NN.hiddenSize)+"node"+".pkl", "wb") as out:
            pickle.dump(NN, out, pickle.HIGHEST_PROTOCOL)
    else:
        with open(activation+"_singlelayer.pkl", "wb") as out:
            pickle.dump(NN, out, pickle.HIGHEST_PROTOCOL)

def nn(filetype,labeltype,get,lsize,nodsize,activation="sigmoid"):
    if get == True:
        NN=getPikle(lsize,nodsize,activation)
    else:
        NN = neuralnetwork.Neural_Network(lsize,nodsize,activation)




    count = 0
    cross_entropy_loss = 0
    for i in range(len(filetype)): # trains the NN 1,000 times
        image = np.reshape(filetype[i],(len(filetype[i]),1)).T
        label = np.reshape(labeltype[i],(len(labeltype[i]),1)).T

        predcit = NN.forward(image)

        """print ("Actual Output:",label )
        print ("Predicted Output:" ,predcit)"""
        if label.argmax(axis=1)==predcit.argmax(axis=1):
            count+=1





    print("Accuracy :",count*100/len(filetype))
    print("hit :", count)

nn(validation_images,validation_labels,True,0,20,activation="relu")
