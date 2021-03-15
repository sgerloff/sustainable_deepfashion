from src.models.triplet_model_factory import TripletModelFactory
import tensorflow as tf
import tensorflow_addons as tfa

from src.data.triplet_dataset_factory import AugmentedTripletDatasetFactory
from src.utility import get_project_dir
import joblib, os

class EfficientNetB0TripletModelFactory(TripletModelFactory):
    def __init__(self, input_shape=(224, 224, 3), embedding_size=128, extraction_layers_size=[1024, 512, 256]):
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.extraction_layers_size = extraction_layers_size
        super().__init__()

    def set_basemodel(self):
        efficient_net = tf.keras.applications.EfficientNetB0(input_shape=self.input_shape,
                                                             include_top=False,
                                                             weights="imagenet")
        efficient_net.trainable = False  # Freeze basemodel by default (transfer learning)
        return efficient_net

    def set_embedding_model(self):
        # See documentation in tensorflow: https://www.tensorflow.org/addons/tutorials/losses_triplet
        embedding_model = tf.keras.models.Sequential(name="embedding_extraction")
        # Reducing dimensions (7,7,1280)->(1,1,1280) to keep parameters in check
        # This mimics the architectore of the original top layers from the EfficientNetb0
        embedding_model.add(tf.keras.layers.GlobalAveragePooling2D())
        embedding_model.add(tf.keras.layers.Flatten())

        # Adding Dense layers according to the extraction layer sizes
        for layer_size in self.extraction_layers_size:
            embedding_model.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation on final dense layer
        embedding_model.add(tf.keras.layers.Dense(self.embedding_size, activation=None))
        # L2 normalize embeddings. This enforces that embedding vectors are on a hypersphere and only the direction matters.
        embedding_model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)))

        return embedding_model

    def preprocessor(self):
        return tf.keras.applications.efficientnet.preprocess_input


if __name__ == "__main__":
    train_df = joblib.load(os.path.join(get_project_dir(),
                                        "data",
                                        "processed",
                                        "category_id_1_deepfashion_train.joblib"))

    model_factory = EfficientNetB0TripletModelFactory()
    model = model_factory.get_model()
    model.compile(
        loss=tfa.losses.TripletSemiHardLoss(),
        optimizer="adam"
    )

    dataset_factory = AugmentedTripletDatasetFactory(train_df)
    train_dataset = dataset_factory.get_dataset(batch_size=16, data_slice_ratio=0.1, shuffle=True)

    model.fit(train_dataset, epochs=2)
    model.summary()