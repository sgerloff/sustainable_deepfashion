from src.models.triplet_model_factory import TripletModelFactory
import tensorflow as tf


class SimpleConv2DTripletModelFactory(TripletModelFactory):
    def __init__(self, input_shape=(224, 224, 3), embedding_size=128, filters_per_conv_layer=[16, 32, 64, 128, 256],
                 size_dense_layers=[512, 256]):
        self.embedding_size = embedding_size
        # Basemodel Layers
        self.filters_per_conv_layer = filters_per_conv_layer
        self.filter_size = 5

        # Embedding Extraction Layers:
        self.size_dense_layers = size_dense_layers

        super().__init__(input_shape=input_shape)

    def set_basemodel(self):
        basemodel = tf.keras.models.Sequential()
        basemodel.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))
        self.set_basemodel_conv_body(basemodel)
        return basemodel

    def set_basemodel_conv_body(self, basemodel):
        for i in range(len(self.filters_per_conv_layer)):
            basemodel.add(tf.keras.layers.Conv2D(self.filters_per_conv_layer[i], self.filter_size, activation="relu"))
            if i + 1 < len(self.filters_per_conv_layer):
                basemodel.add(tf.keras.layers.MaxPooling2D())

    def set_embedding_model(self):
        embedding_model = tf.keras.models.Sequential()
        # Flatten with GlobalAveragePooling2D:
        embedding_model.add(tf.keras.layers.GlobalAveragePooling2D())
        # Build dense layers:
        self.set_embedding_dense_layers(embedding_model)

        embedding_model.add(tf.keras.layers.Dense(self.embedding_size, activation="linear"))
        embedding_model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)))
        return embedding_model

    def set_embedding_dense_layers(self, embedding_model):
        for n in self.size_dense_layers:
            embedding_model.add(tf.keras.layers.Dense(n, activation="relu"))

    def preprocessor(self):
        return tf.keras.applications.imagenet_utils.preprocess_input


class SimpleConv2DDropoutAndBatchNormalizationTripletModelFactory(SimpleConv2DTripletModelFactory):
    def __init__(self,
                 input_shape=(224, 224, 3),
                 embedding_size=128,
                 filters_per_conv_layer=[16, 32, 64, 128, 256],
                 batch_norm_bool=[False, False, False, False, False],
                 size_dense_layers=[512, 256],
                 dense_dropout_ratios=[0.5, 0.5]
                 ):
        self.dense_dropout_ratios = dense_dropout_ratios
        self.batch_norm_bool = batch_norm_bool

        super().__init__(
            input_shape=input_shape,
            embedding_size=embedding_size,
            filters_per_conv_layer=filters_per_conv_layer,
            size_dense_layers=size_dense_layers
        )

        assert len(self.batch_norm_bool) == len(self.filters_per_conv_layer)
        assert len(self.dense_dropout_ratios) == len(self.size_dense_layers)

    def set_basemodel_conv_body(self, basemodel):
        for i in range(len(self.filters_per_conv_layer)):
            basemodel.add(tf.keras.layers.Conv2D(self.filters_per_conv_layer[i], self.filter_size))
            if self.batch_norm_bool[i]:
                basemodel.add(tf.keras.layers.BatchNormalization())
            basemodel.add(tf.keras.layers.ReLU())
            if i + 1 < len(self.filters_per_conv_layer):
                basemodel.add(tf.keras.layers.MaxPooling2D())

    def set_embedding_dense_layers(self, embedding_model):
        for i in range(len(self.size_dense_layers)):
            embedding_model.add(tf.keras.layers.Dense(self.size_dense_layers[i], activation="relu"))
            embedding_model.add(tf.keras.layers.Dropout(self.dense_dropout_ratios[i]))


if __name__ == "__main__":
    simple_conv_factory = SimpleConv2DTripletModelFactory()
    model = simple_conv_factory.get_model()
    model.compile(
        loss="mse",
        optimizer="adam"
    )
    print(model.layers[1].summary())
    print(model.layers[2].summary())
