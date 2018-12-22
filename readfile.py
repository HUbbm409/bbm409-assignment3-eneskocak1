from scipy.io import loadmat
import numpy as np
import pickle
import os
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


def getPikle(path):
    with open(path+".pkl", "rb") as inp:
        NN = pickle.load(inp)
    return NN

def setPikle(NN,path):

    with open(path+".pkl", "wb") as out:
        pickle.dump(NN, out, pickle.HIGHEST_PROTOCOL)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)