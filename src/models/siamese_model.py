from tensorflow.keras.applications.efficientnet import EfficientNetB7
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

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
    model = get_siamese_model((600,600,3))
    print(model.summary())