from readfile import getImageInfo,getPikle
import numpy as np


import time

train_images, train_labels = getImageInfo("train.mat")
test_images, test_labels = getImageInfo("test.mat")
validation_images, validation_labels = getImageInfo("validation.mat")



def nntest_valid(filetype,labeltype,lsize,nodsize,activation="sigmoid"):
    NN=getPikle(lsize,nodsize,activation)
    count = 0
    cross_entropy_loss = 0


    for i in range(len(filetype)): # trains the NN 1,000 times
        image = np.reshape(filetype[i],(len(filetype[i]),1)).T
        label = np.reshape(labeltype[i],(len(labeltype[i]),1)).T

        predcit = NN.forward(image)
        loss= NN.cross_entropy(predcit, label)
        cross_entropy_loss += loss

        #print ("Actual Output:",label )
        #print ("Predicted Output:" ,predcit)
        if label.argmax(axis=1)==predcit.argmax(axis=1):
            count+=1

    print("Accuracy :",count*100/len(filetype))
    print("hit :", count)
    print("loss: ",cross_entropy_loss/len(filetype))

    return count*100/len(filetype),cross_entropy_loss/len(filetype)

