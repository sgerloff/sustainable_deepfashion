from src.models.triplet_model import TripletModel
from src.data.triplet_batch_generator import TripletBatchGenerator
from src.metrics.triplet_metrics import PessimisticTripletMetric
from src.utility import get_project_dir

import tensorflow as tf
import tensorflow_addons as tfa

import joblib
import json
import os
import time


def get_dataset(generator, batch_size=32, training_size=100, resolution=224):
    dataset = tf.data.Dataset.from_generator(generator, args=[batch_size, training_size],
                                             output_types=(tf.float16, tf.float16),
                                             output_shapes=([resolution, resolution, 3], ())
                                             )
    return dataset.cache().batch(batch_size, drop_remainder=True).prefetch(2).repeat()


if __name__ == "__main__":
    train_dataframe = joblib.load(os.path.join(get_project_dir(),
                                               "data",
                                               "processed",
                                               "category_id_1_deepfashion_train.joblib"))

    resolution = 224
    triplet_batch = TripletBatchGenerator(train_dataframe, resolution)

    training_size = train_dataframe["pair_id"].nunique() // 100
    batch_size = 64

    dataset = get_dataset(triplet_batch.tf_generator,
                          batch_size=batch_size,
                          training_size=training_size,
                          resolution=resolution)

    pessimistic_metric = PessimisticTripletMetric()

    model = TripletModel((resolution, resolution, 3), trainable=True)
    model.compile(loss=tfa.losses.TripletSemiHardLoss(),
                  optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                  metrics=[pessimistic_metric.score])
    history = model.fit(dataset,
                        steps_per_epoch=training_size,
                        epochs=10,
                        callbacks=[tf.keras.callbacks.ModelCheckpoint(
                            os.path.join(get_project_dir(), "models", "triplet_" + time.strftime("%Y%m%d") + ".h5")
                        )])

    json.dump(history, open(os.path.join(get_project_dir(),
                                         "reports",
                                         "triplet_history_" + time.strftime("%Y%m%d") + ".json")))
