from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf


class TripletBatchGenerator:
    def __init__(self, database, resolution):
        self.df = database
        self.resolution = resolution
        self.batch_size = 4

    def get_array_for_item(self, i):
        # path = os.path.join(self.df.loc[i, "image"])
        x = tf.io.read_file(self.df.loc[i, "image"])
        x = tf.io.decode_image(x)
        x = tf.image.resize(x, [self.resolution, self.resolution])
        return preprocess_input(x)

    def generate_batch_for_pair_id(self, pair_id, batch_size):
        map = self.df["pair_id"] == pair_id
        positive = self.df[map]["image"]
        negative = self.df[~map]["image"]

        batch_indices = []

        if len(positive) < batch_size // 2:
            batch_indices.extend(positive.index)
        else:
            batch_indices.extend(positive.sample(batch_size // 2).index)

        remainder = batch_size - len(batch_indices)
        batch_indices.extend(negative.sample(remainder).index)
        return batch_indices

    def tf_generator(self, bs, training_range):
        unique_pair_ids = self.df["pair_id"].unique()

        for i in range(training_range):
            pair_id = unique_pair_ids[i % len(unique_pair_ids)]
            batch_indices = self.generate_batch_for_pair_id(pair_id, bs)
            for b in batch_indices:
                yield self.get_array_for_item(b), self.df.loc[b, "pair_id"]

    def get_keras_dataset(self, batch_size=64, training_size_ratio=1.):
        training_size = int(training_size_ratio*len(self.df["pair_id"].unique()))
        dataset = tf.data.Dataset.from_generator(self.tf_generator, args=[batch_size, training_size],
                                                 output_types=(tf.float16, tf.float16),
                                                 output_shapes=([self.resolution, self.resolution, 3], ())
                                                 )
        return dataset.cache().batch(batch_size, drop_remainder=True).prefetch(2).repeat(), training_size
