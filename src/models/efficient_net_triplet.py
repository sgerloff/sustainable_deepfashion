from tensorflow.keras.applications.efficientnet import EfficientNetB0
import tensorflow as tf
import tensorflow_addons as tfa

from src.metrics.triplet_metrics import PessimisticTripletMetric
from src.data.triplet_batch_generator import TripletBatchGenerator

from tensorflow.keras import layers
from tensorflow.keras import models

from src.utility import get_project_dir

import time, os


class EfficientNetTriplet:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.basemodel = None
        self.metric = PessimisticTripletMetric()

        self.define_model()

    def define_model(self):
        input_image = layers.Input(self.input_shape)

        # Pretrained model
        self.basemodel = EfficientNetB0(input_tensor=input_image, include_top=False, weights="imagenet")
        self.basemodel.trainable = False
        x = self.basemodel(input_image)

        embedding_model = self.set_embedding_model()

        x = embedding_model(x)

        self.model = models.Model(inputs=input_image, outputs=x)
        self.compile()
        return self.model

    def set_embedding_model(self):
        # See documentation in tensorflow: https://www.tensorflow.org/addons/tutorials/losses_triplet
        embedding_model = models.Sequential(name="embedding_extraction")
        # Reducing dimensions (7,7,1280)->(1,1,1280) to keep parameters in check
        embedding_model.add(layers.GlobalAveragePooling2D())
        embedding_model.add(layers.Flatten())
        # No activation on final dense layer
        embedding_model.add(layers.Dense(256, activation=None))
        # L2 normalize embeddings
        embedding_model.add(layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)))

        return embedding_model

    def set_basemodel_freeze_ratio(self, ratio):
        # Setting the layers trainable parameter does not reset the basemodel's parameter
        self.basemodel.trainable = True

        ratio_index = int(ratio * len(self.basemodel.layers))

        for layer in self.basemodel.layers[:ratio_index]:
            layer.trainable = False
        for layer in self.basemodel.layers[ratio_index:]:
            layer.trainable = True

        # Recompile the model to make sure the changes are applied
        self.compile()

        return self.model

    def compile(self):
        self.model.compile(
            loss=tfa.losses.TripletSemiHardLoss(),
            optimizer=tf.optimizers.Adam(learning_rate=1e-5),
            metrics=[self.metric.score]
        )

        return self.model

    def set_learning_rate(self, learning_rate):
        tf.keras.backend.set_value(self.model.optimizer.lr, learning_rate)

    def train(self,
              train_dataframe,
              validation_dataframe,
              epochs=10,
              training_ratio=0.1,
              batch_size=64,
              callbacks=None):
        train, train_tsize = self.get_dataset(train_dataframe,
                                              training_ratio=training_ratio,
                                              batch_size=batch_size)
        validation, validation_tsize = self.get_dataset(validation_dataframe,
                                                        training_ratio=training_ratio,
                                                        batch_size=batch_size)

        history = self.model.fit(train,
                                 steps_per_epoch=train_tsize,
                                 epochs=epochs,
                                 validation_data=validation,
                                 validation_steps=validation_tsize,
                                 callbacks=callbacks)
        # [tf.keras.callbacks.ModelCheckpoint(
        #     os.path.join(get_project_dir(), "models",
        #                  "triplet_" + time.strftime("%Y%m%d") + ".h5")
        # )]
        return history

    def get_dataset(self, dataframe, training_ratio=0.1, batch_size=64):
        triplet_batch = TripletBatchGenerator(dataframe, self.input_shape[0])
        return triplet_batch.get_keras_dataset(training_size_ratio=training_ratio, batch_size=batch_size)

    def save(self, file, fileName=True):
        if fileName:
            file = self.get_default_path(file)
        print(f"Save model to: {file}")
        tf.keras.models.save_model(self.model, file)

    def load(self, file, fileName=True):
        if fileName:
            file = self.get_default_path(file)
        dependencies = {
            "score": self.metric.score
        }
        print(f"Load model from {file}")
        self.model = tf.keras.models.load_model(file, custom_objects=dependencies)
        self.basemodel = self.model.layers[1]

    @staticmethod
    def get_default_path(file):
        return os.path.join(get_project_dir(), "models", file)


class EfficientNetTripletBigTop(EfficientNetTriplet):
    def set_embedding_model(self):
        # See documentation in tensorflow: https://www.tensorflow.org/addons/tutorials/losses_triplet
        embedding_model = models.Sequential(name="embedding_extraction")
        embedding_model.add(layers.GlobalAveragePooling2D())
        embedding_model.add(layers.Dense(512, activation="relu"))
        # No activation on final dense layer
        embedding_model.add(layers.Dense(256, activation=None))
        # L2 normalize embeddings
        embedding_model.add(layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)))

        return embedding_model


if __name__ == "__main__":
    tmodel = EfficientNetTriplet()
    tmodel.set_basemodel_freeze_ratio(0.5)
    print(tmodel.model.summary())

    tmodel.save("triplet_test_" + time.strftime("%Y%m%d")+".h5")
    tmodel.load("triplet_test_" + time.strftime("%Y%m%d")+".h5")

    tmodel2 = EfficientNetTripletBigTop()
    print(tmodel2.model.layers[2].summary())