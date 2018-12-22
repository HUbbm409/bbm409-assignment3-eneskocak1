from readfile import getPikle,getImageInfo
import numpy as np
import argparse
import pickle

def nntest_valid(filetype,labeltype,modelpath):
    NN=getPikle(modelpath)
    count = 0
    cross_entropy_loss = 0
    for i in range(len(filetype)): # trains the NN 1,000 times
        image = np.reshape(filetype[i],(len(filetype[i]),1)).T
        label = np.reshape(labeltype[i],(len(labeltype[i]),1)).T

        predcit = NN.forward(image)
        loss= NN.cross_entropy(predcit, label)
        cross_entropy_loss += loss

        if label.argmax(axis=1)==predcit.argmax(axis=1):
            count+=1
    print("\nTest/Validation:")
    print("Accuracy :",count*100/len(filetype))
    print("hit :", count)
    print("loss: ",cross_entropy_loss/len(filetype))

    return count*100/len(filetype),cross_entropy_loss/len(filetype)


def test_main(filetype,labeltype,modelpath):

    with open(modelpath, "rb") as inp:
        NN = pickle.load(inp)
    count = 0
    cross_entropy_loss = 0
    for i in range(len(filetype)): # trains the NN 1,000 times
        image = np.reshape(filetype[i],(len(filetype[i]),1)).T
        label = np.reshape(labeltype[i],(len(labeltype[i]),1)).T
        predcit = NN.forward(image)
        loss= NN.cross_entropy(predcit, label)
        cross_entropy_loss += loss

        if label.argmax(axis=1)==predcit.argmax(axis=1):
            count+=1
    print("\nTest/Validation:")
    print("Accuracy :",count*100/len(filetype))
    print("hit :", count)
    print("loss: ",cross_entropy_loss/len(filetype))

    return count*100/len(filetype),cross_entropy_loss/len(filetype)
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-data_path", required=True,
                    help="path to train data. format must be .mat")
    ap.add_argument("-model_path", required=True,
                    help="path to trained model must be  .pkl")
    args = vars(ap.parse_args())

    test_images, test_labels = getImageInfo(args["data_path"])
    test_main(test_images,test_labels,args["model_path"])