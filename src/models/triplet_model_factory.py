import tensorflow as tf


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
        """
        Freezes the first 100*ratio percent of the layers
        of the basemodel. The remaining layers are set to
        trainable. BatchNormalization layers are frozen to
        prevent loss of pretrained weights.
        """

        self.basemodel.trainable = True

        ratio_index = int(ratio * len(self.basemodel.layers))

        for layer in self.basemodel.layers[:ratio_index]:
            layer.trainable = False
        for layer in self.basemodel.layers[ratio_index:]:
            if layer.__class__.__name__ == "BatchNormalization":
                layer.trainable = False
            else:
                layer.trainable = True

    def get_model(self):
        input_image = tf.keras.layers.Input(self.input_shape)
        x = self.basemodel(input_image)
        x = self.embedding_model(x)

        return tf.keras.models.Model(inputs=input_image, outputs=x)

    def preprocessor(self):
        return lambda x: x
