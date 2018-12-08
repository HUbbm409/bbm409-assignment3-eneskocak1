from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def getImageInfo(filepath):

    images = loadmat(filepath)
    imagelist = np.array(images["x"]/255)
    labels =np.array(images["y"][0])
    classes=len(set(labels))
    print(labels[0])
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





    """for i in range(len(imagelist)):
        image = np.reshape(imagelist[i], (32, 24))
        plt.imshow(image, cmap=plt.get_cmap('gray'))
        print(labellist[i])
        plt.show()
        input("asd")"""

    return imagelist,labellist