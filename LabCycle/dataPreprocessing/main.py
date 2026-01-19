
import os

from loader import load_images,load_labels
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
if __name__=="__main__":
    x_train=load_images("/home/jd/NextWordPrediction/LabCycle/archive/train-images.idx3-ubyte")
    y_train=load_labels("/home/jd/NextWordPrediction/LabCycle/archive/train-labels.idx1-ubyte")

    x_test=load_images("/home/jd/NextWordPrediction/LabCycle/archive/t10k-images.idx3-ubyte")
    y_test=load_labels("/home/jd/NextWordPrediction/LabCycle/archive/train-labels.idx1-ubyte")


    #normalize

    x_train=x_train/255
    x_test=x_test/255
    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_oh = encoder.transform(y_test.reshape(-1, 1))

    # Train-validation split
    X_train, X_val, y_train_oh, y_val_oh = train_test_split(
        x_train,
        y_train_oh,
        test_size=0.2,
        random_state=42
    )
    # Print shapes
    print("Training set:", X_train.shape, y_train_oh.shape)
    print("Validation set:", X_val.shape, y_val_oh.shape)
    print("Test set:", x_test.shape, y_test_oh.shape)