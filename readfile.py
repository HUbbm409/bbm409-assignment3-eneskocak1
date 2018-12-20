from scipy.io import loadmat
import numpy as np
import pickle

def getImageInfo(filepath):

    images = loadmat(filepath)
    imagelist = np.array(images["x"]/255)
    labels =np.array(images["y"][0])
    classes=len(set(labels))

    labellist=[]
    for i in labels:
        hot = []
        for k in range(classes):
            if k==i:
                hot.append(1)
            else:
                hot.append(0)
        labellist.append(hot)

    labellist=np.array(labellist)



    return imagelist,labellist


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