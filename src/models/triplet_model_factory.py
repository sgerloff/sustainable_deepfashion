import tensorflow as tf
from src.utility import savely_unfreeze_layers_of_model

class TripletModelFactory:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.basemodel = self.set_basemodel()
        self.embedding_model = self.set_embedding_model()

    def set_basemodel(self):
        pass

    def set_embedding_model(self):
        pass

    def set_basemodel_freeze_ratio(self, ratio):
        self.basemodel = savely_unfreeze_layers_of_model(self.basemodel, ratio)

    def get_model(self):
        input_image = tf.keras.layers.Input(self.input_shape)
        x = self.basemodel(input_image)
        x = self.embedding_model(x)

        return tf.keras.models.Model(inputs=input_image, outputs=x)

    def preprocessor(self):
        return lambda x: x
