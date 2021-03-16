from src.models.triplet_model_factory import TripletModelFactory
import tensorflow as tf


class SimpleConv2DTripletModelFactory(TripletModelFactory):
    def __init__(self, input_shape=(224, 224, 3), embedding_size=128, filters_per_conv_layer=[16, 32, 64, 128, 256],
                 size_dense_layers=[512, 256]):
        self.embedding_size = embedding_size
        # Basemodel Layers
        self.filters_per_conv_layer = filters_per_conv_layer
        self.basemodel_layer_dict = {}
        self.build_basemodel_layer_dict()

        # Embedding Extraction Layers:
        self.size_dense_layers = size_dense_layers
        self.embedding_layer_dict = {}
        self.build_embedding_layer_dict()

        super().__init__(input_shape=input_shape)

    def build_basemodel_layer_dict(self):
        for i, f in enumerate(self.filters_per_conv_layer):
            self.basemodel_layer_dict["conv_" + str(i)] = tf.keras.layers.Conv2D(f, 5, activation="relu")
            if i + 1 < len(self.filters_per_conv_layer):
                self.basemodel_layer_dict["pool_" + str(i)] = tf.keras.layers.MaxPooling2D()

    def set_basemodel(self):
        input = tf.keras.layers.Input(shape=self.input_shape)
        output = tf.keras.layers.InputLayer()(input)
        for i in range(len(self.filters_per_conv_layer)):
            if i + 1 < len(self.filters_per_conv_layer):
                output = self.basemodel_layer_dict["conv_" + str(i)](output)
                output = self.basemodel_layer_dict["pool_" + str(i)](output)
            else:
                output = self.basemodel_layer_dict["conv_" + str(i)](output)
        return tf.keras.models.Model(inputs=input, outputs=output, name="basemodel_conv2d")

    def build_embedding_layer_dict(self):
        """
        Build embedding extractions model:

        1. The model starts with a GlobalAveragePooling, to reduce
        and flatten the output of the convolutional layers of the basemodel.

        2. Apply dense layers according to "self.size_dense_layers"

        3. Apply final dense layer with linear activation to create embedding
        vector of size "self.embedding_size" and normalize with l2-norm
        """
        self.embedding_layer_dict["global_avg_pool_0"] = tf.keras.layers.GlobalAveragePooling2D()
        # Build dense layers:
        for i, n in enumerate(self.size_dense_layers):
            self.embedding_layer_dict["dense_" + str(i)] = tf.keras.layers.Dense(n, activation="relu")
        # Create and normalize embedding:
        self.embedding_layer_dict["embedding_output_dense"] = tf.keras.layers.Dense(self.embedding_size,
                                                                                    activation="linear")
        self.embedding_layer_dict["embedding_output_norm"] = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.l2_normalize(x, axis=1))

    def set_embedding_model(self):
        # Retrieve shape of output of basemodel:
        output_shape_basemodel = self.basemodel.layers[-1].output_shape[1:]
        # Define input:
        input = tf.keras.layers.Input(shape=output_shape_basemodel)
        # Flatten with GlobalAveragePooling2D:
        output = self.embedding_layer_dict["global_avg_pool_0"](input)
        # Run through dense layers:
        for i in range(len(self.size_dense_layers)):
            output = self.embedding_layer_dict["dense_" + str(i)](output)
        # Create and normalize embedding vectors (= output)
        output = self.embedding_layer_dict["embedding_output_dense"](output)
        output = self.embedding_layer_dict["embedding_output_norm"](output)
        return tf.keras.models.Model(inputs=input, outputs=output, name="extraction_dense")


if __name__ == "__main__":
    simple_conv_factory = SimpleConv2DTripletModelFactory()
    model = simple_conv_factory.get_model()
    model.compile(
        loss="mse",
        optimizer="adam"
    )
    print(model.layers[1].summary())
    print(model.layers[2].summary())
