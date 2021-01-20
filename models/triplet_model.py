from tensorflow.keras.applications.efficientnet import EfficientNetB0
import numpy as np

from tensorflow.keras.layers import Input
from keras.models import Sequential
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import Model


def TripletModel(input_shape):
    input_image = Input(input_shape)
    effnet = EfficientNetB0(input_tensor=input_image, include_top=False, weights="imagenet")
    effnet.trainable = False
    x = effnet(input_image)

    embedding_model = Sequential()

    # #Alterantive to reducing the dimensions properly...
    # #Reduce the output size to the desired embedding size 7x7x1280 -> 1x1x128
    # embedding_model.add(layers.Conv2D(512, kernel_size=1, activation="relu"))
    # embedding_model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
    # embedding_model.add(layers.Conv2D(256, kernel_size=1, activation="relu"))
    # embedding_model.add(layers.MaxPool2D(pool_size=(2,2), strides=2))
    # embedding_model.add(layers.Conv2D(128, kernel_size=1, activation="relu"))
    # #Flatten the embedding and normalize to the L2-Sphere
    # embedding_model.add(layers.Flatten())
    # embedding_model.add(layers.Lambda(lambda t: K.l2_normalize(t)))

    # Inspired by: https://developer.ridgerun.com/wiki/index.php?title=GstInference/Supported_architectures/FaceNet
    embedding_model.add(layers.AveragePooling2D(pool_size=(7, 7)))
    embedding_model.add(layers.Flatten())
    embedding_model.add(layers.Dense(128, activation="relu"))
    embedding_model.add(layers.Softmax())

    x = embedding_model(x)

    triplet_model = Model(inputs=input_image, outputs=x)
    return triplet_model


if __name__ == "__main__":
    model = TripletModel((224, 224, 3))
    sample_input = np.random.randn(1, 224, 224, 3)
    model.compile(loss="TripletSemiHardLoss", optimizer="adam")
    print(model.summary())

    sample_input = np.random.randn(10, 224, 224, 3)
    print(sample_input.shape)
    output = model.predict(sample_input)
    print(output.shape)
