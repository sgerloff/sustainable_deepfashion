import numpy as np
import joblib

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB7
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from batch_generator import BatchGenerator

def get_siamese_model(input_shape):
    """
        Model architecture
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Pretrained Convolutional Neural Network
    effnet = EfficientNetB7(weights='imagenet')
    for layer in effnet.layers[:-1]:
        layer.trainable = False
    effnet.layers[-1].trainable = True

    # Generate the encodings (feature vectors) for the two images
    encoded_l = effnet(left_input)
    encoded_r = effnet(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net


if __name__ == "__main__":


    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print(f"Load database...")
    df = joblib.load("deepfashion_train.joblib")
    print(f"Load training pairs...")
    pairs = joblib.load("pairs_training.joblib")

    df_val = joblib.load("deepfashion_validation.joblib")
    pairs_val = joblib.load("pairs_validation.joblib")

    batch_generator = BatchGenerator(df, pairs)
    val_generator = BatchGenerator(df_val, pairs_val)

    model = get_siamese_model((600, 600, 3,))

    opt = tf.keras.optimizers.Adam(1e-4)
    #The latter is needed to use full memory of the gpu (RTX 2080), which has dedicated memory for float only.
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics="accuracy")
    model.fit(batch_generator.generate(batch_size=4, train_range=160),
              steps_per_epoch=16,
              epochs=10,
              # validation_data=val_generator.generate(batch_size=4, train_range=160),
              # validation_steps=16,
              # verbose=2
              )
    # model.save("test_model")

