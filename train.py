import pandas as pd
import numpy as np
import joblib
import random

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet import decode_predictions
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

def get_array_for_item(i):
    im = Image.open(df.loc[i,"image"])
    bounding = df.loc[i,"bounding_box"]
    imc = im.crop( ( bounding[0], bounding[1], bounding[2], bounding[3] ) ).resize((600,600))
    x = image.img_to_array(imc)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)

def get_batch(batch_size, offset=0):
    X_pairs = [ np.empty((batch_size, 600, 600, 3)) for i in range(2) ]
    targets = np.zeros((batch_size,1))

    targets[:batch_size//2] = 1
    for i in range(batch_size):
        if i < batch_size//2:
            X_pairs[0][i] = get_array_for_item(pairs[offset+i][0])
            X_pairs[1][i] = get_array_for_item(pairs[offset+i][1])
        else:
            X_pairs[0][i] = get_array_for_item(negative_pairs[offset+i][0])
            X_pairs[1][i] = get_array_for_item(negative_pairs[offset+i][1])
    return X_pairs, targets

def batch_generator(batch_size):
    for i in range(len(pairs)//batch_size):
        p, t = get_batch(batch_size, offset=i*batch_size)
        yield (p, t)

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
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # return the model
    return siamese_net

if __name__ == "__main__":
    df = joblib.load("deepfashion_train.joblib")

    cat1 = df[df["category_id"]==1]
    unique_pair_id = cat1["pair_id"].unique()

    #Generate positive examples:
    pairs = []
    for u in unique_pair_id[:100]:
        tmp = list(cat1[cat1["pair_id"]==u].index)
        for i in range(len(tmp)-1):
            a = tmp[i]
            for b in tmp[i+1:]:
                pairs.append((a,b))
    print(len(pairs))
    random.shuffle(pairs)

    #Generate negative examples:
    tmp = cat1[cat1["pair_id"] < unique_pair_id[100]].index
    negative_pairs = []
    while len(negative_pairs) < len(pairs):
        t = random.choices(tmp, k=2)
        if t[0] != t[1] and cat1.loc[t[0],"pair_id"] != cat1.loc[t[1], "pair_id"]:
            negative_pairs.append(tuple(t))
    print(len(negative_pairs))
    random.shuffle(negative_pairs)

    model = get_siamese_model((600,600,3,))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
    batch_size=4
    model.fit( batch_generator(batch_size), steps_per_epoch=len(pairs)//batch_size, epochs=10 )
    model.save("test_model")


