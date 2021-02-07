from models.triplet_model import TripletModel
from triplet_batch import TripletBatch

import tensorflow as tf
import tensorflow_addons as tfa

from metrics.triplet_metrics import OptimisticTripletMetric, PessimisticTripletMetric, AllTripletMetric

import joblib

resolution = 224

df = joblib.load("category_id_1_deepfashion_train.joblib")
print(df.image)

triplet_batch = TripletBatch(df, resolution)

training_size = df["pair_id"].nunique() // 1
bs = 32
dataset = tf.data.Dataset.from_generator(triplet_batch.tf_generator, args=[bs, training_size],
                                         output_types=(tf.float16, tf.float16),
                                         output_shapes=([resolution, resolution, 3], ())
                                         )

dataset = dataset.batch(bs, drop_remainder=True).repeat()

# metrics = [OptimisticTripletMetric(), PessimisticTripletMetric(), AllTripletMetric()]
metric = PessimisticTripletMetric()

model = TripletModel((resolution, resolution,3), trainable=True)
model.compile(loss=tfa.losses.TripletSemiHardLoss(),
              optimizer="adam",
              metrics=[metric.score])
model.fit(dataset, steps_per_epoch=training_size, epochs=10)

# print(list(dataset.take(1))[1])