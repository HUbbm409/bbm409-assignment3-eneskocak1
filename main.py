from readfile import getImageInfo
from train import nn
import matplotlib.pyplot as plt

train_images, train_labels = getImageInfo("train.mat")
validation_images, validation_labels = getImageInfo("validation.mat")



def main():

    loss,epoch,accuracy,valid_loss,valid_accuracy=nn( epoch=300,
                            filetype=train_images,
                            labeltype=train_labels,
                            pickle=False,
                            lsize =1,
                            nodsize=100,
                            activation="sigmoid",
                            batch=1,
                            alpha=0.02)

    plt.title("Schoastic Gradient Descent")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.plot(epoch, loss, color="blue", label="train", linewidth=1, linestyle="-")
    plt.plot(epoch, valid_loss, color="red", label="validation", linewidth=1, linestyle="-")



    plt.legend(loc='lower left')

    plt.show()


main()