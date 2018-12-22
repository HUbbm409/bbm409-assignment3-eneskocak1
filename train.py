from readfile import *
import numpy as np
import neuralnetwork
from test import nntest_valid
import matplotlib.pyplot as plt
import cv2
import os
import argparse






def nn(epoch=100,filetype =None,labeltype= None,pickle = False ,lsize = 0,nodsize=0,activation="sigmoid",batch=1,alpha=1,path=None,modelpath=None):
    if pickle == True:
        NN=getPikle(modelpath)
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

        setPikle(NN,modelpath)

        print("\nTrain:")
        print("epoch:", k)
        print("Accuracy :", count * 100 / len(filetype))
        print("hit :", count)
        print("loss:", cross_entropy_loss / len(filetype))

        train_loss.append(cross_entropy_loss/len(filetype))
        train_accuracy.append(count*100/len(filetype))
        train_epoch.append(k)

        validation_images, validation_labels = getImageInfo("validation.mat")
        validaccuracy,validloss=nntest_valid(validation_images, validation_labels,modelpath)
        valid_loss.append(validloss)
        valid_accuracy.append(validaccuracy)
    if lsize ==0:
        for i  in range(labeltype.shape[1]):
            image =np.reshape(NN.W[lsize].T[i],(32,24))
            image = np.abs(image)
            cv2.imwrite(os.path.join(path,"activation_"+activation+"_layer"+str(lsize)+"_node"+str(nodsize)+"_out"+str(i+1)+".jpg"),image*255)


    return train_loss,train_epoch,train_accuracy,valid_loss,valid_accuracy


def train(epoch=100,filetype =None,labeltype= None,pickle = False ,lsize = 0,nodsize=0,activation="sigmoid",batch=1,alpha=1):
    if batch == 1:
        batchtype = "Schoastic"
    else:
        batchtype = "Batch :" + str(batch)
    if lsize == 0:
        layer = "[Activation: " + activation + " , Learnin rate: " + str(alpha) + ", Singlelayer ]"
    else:
        layer = "[Activation: " + activation + " , Learnin rate: " + str(alpha) + ", Multilayer(" + str(
            lsize) + "layer," + str(nodsize) + "node) ]"
    modelpath = "models/" + batchtype + layer

    if pickle == True:
        NN=getPikle(modelpath)
    else:
        NN = neuralnetwork.Neural_Network(lsize,nodsize,activation,filetype.shape[1],labeltype.shape[1])
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

        setPikle(NN,modelpath)

        print("\nTrain:")
        print("epoch:", k)
        print("Accuracy :", count * 100 / len(filetype))
        print("hit :", count)
        print("loss:", cross_entropy_loss / len(filetype))


def results(epoch=100,pickle = False ,lsize = 0,nodsize=0,activation="sigmoid",batch=1,alpha=1):
    filetype, labeltype = getImageInfo("train.mat")

    if batch == 1:
        batchtype = "Schoastic"
    else:
        batchtype = "Batch :" + str(batch)
    if lsize == 0:
        path = "singlelayer_results/"
        layer = "[Activation: " + activation + " , Learnin rate: " + str(alpha) + ", Singlelayer ]"
    else:
        path= "multilayer_results/"
        layer = "[Activation: " + activation + " , Learnin rate: " + str(alpha) + ", Multilayer(" + str(
            lsize) + "layer," + str(nodsize) + "node) ]"
    moedelpath="models/"+batchtype+layer
    path+= batchtype+layer
    createFolder(path)
    loss,epoch,accuracy,valid_loss,valid_accuracy=nn( epoch=epoch,
                            filetype=filetype,
                            labeltype=labeltype,
                            pickle=pickle,
                            lsize =lsize,
                            nodsize=nodsize,
                            activation=activation,
                            batch=batch,
                            alpha=alpha,
                            path=path,
                            modelpath=moedelpath)


    plt.figure(figsize=(20, 10))
    plt.suptitle(batchtype+" Gradient Descent with \n"+
                 layer)
    plt.subplot(1, 2, 1)
    plt.title("Train-Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Avarage Loss")
    plt.plot(epoch, loss, color="blue", label="train", linewidth=1, linestyle="-")
    plt.plot(epoch, valid_loss, color="red", label="validation", linewidth=1, linestyle="-")
    plt.legend(loc='lower left')

    plt.subplot(1, 2, 2)
    plt.title("Train-Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epoch, accuracy, color="blue", label="train", linewidth=1, linestyle="-")
    plt.plot(epoch, valid_accuracy, color="red", label="validation", linewidth=1, linestyle="-")
    plt.legend(loc='lower left')
    plt.savefig(path+"/"+batchtype+layer+".png")
    plt.close()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-data_path", required=True,
                    help="path to train data. format must be .mat")
    args = vars(ap.parse_args())

    train_images, train_labels = getImageInfo(args["data_path"])
    train(epoch=300,
       filetype=train_images,
       labeltype=train_labels,
       pickle=False,
       lsize=2,
       nodsize=100,
       activation="sigmoid",
       batch=1,
       alpha=0.02)



