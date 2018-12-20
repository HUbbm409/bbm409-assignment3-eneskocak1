from readfile import getImageInfo,getPikle,setPikle
import numpy as np
import neuralnetwork
from test import nntest_valid


train_images, train_labels = getImageInfo("train.mat")
validation_images, validation_labels = getImageInfo("validation.mat")



def nn(epoch=100,filetype =None,labeltype= None,pickle = False ,lsize = 0,nodsize=0,activation="sigmoid",batch=1,alpha=1):
    if pickle == True:
        NN=getPikle(lsize,nodsize,activation)
    else:
        NN = neuralnetwork.Neural_Network(lsize,nodsize,activation,filetype.shape[1],labeltype.shape[1])

    train_loss=[]
    train_accuracy=[]
    train_epoch=[]
    valid_loss=[]
    valid_accuracy=[]

    for k in range(epoch):

        count =0
        cross_entropy_loss = 0
        for i in range(len(filetype)): # trains the NN 1,000 times

            image = np.reshape(filetype[i],(len(filetype[i]),1)).T
            label = np.reshape(labeltype[i],(len(labeltype[i]),1)).T

            predcit ,loss= NN.train(image,label,alpha,batch)
            cross_entropy_loss  += loss
            if label.argmax(axis=1)==predcit.argmax(axis=1):
                count+=1

        setPikle(NN,activation)
        train_loss.append(cross_entropy_loss/len(filetype))
        train_accuracy.append(count*100/len(filetype))
        train_epoch.append(k)
        validaccuracy,validloss=nntest_valid(validation_images, validation_labels, lsize, nodsize, activation=activation)
        valid_loss.append(validloss)
        valid_accuracy.append(validaccuracy)

        print("epoch:",k)
        print("Accuracy :",count*100/len(filetype))
        print("hit :", count)
        print("loss:",cross_entropy_loss/len(filetype))

    return train_loss,train_epoch,train_accuracy,valid_loss,valid_accuracy
