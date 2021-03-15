from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import random

from src.utility import get_project_dir
import joblib, os


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
        random.shuffle(unique_pair_ids)  # Don't forget to shuffle the training data.
        for i in range(training_range):
            pair_id = unique_pair_ids[i % len(unique_pair_ids)]
            batch_indices = self.generate_batch_for_pair_id(pair_id, bs)
            for b in batch_indices:
                yield self.get_array_for_item(b), self.df.loc[b, "pair_id"]

    def get_keras_dataset(self, batch_size=64, training_size_ratio=1.):
        training_size = int(training_size_ratio * len(self.df["pair_id"].unique()))
        dataset = tf.data.Dataset.from_generator(self.tf_generator, args=[batch_size, training_size],
                                                 output_types=(tf.float16, tf.float16),
                                                 output_shapes=([self.resolution, self.resolution, 3], ())
                                                 )
        return dataset.batch(batch_size, drop_remainder=True).prefetch(1).repeat(), training_size


class GenericTripletBatchGenerator:
    def __init__(self, database):
        self.df = database
        self.unique_pair_ids = self.df["pair_id"].unique()

    def tf_generator(self, batch_size, data_slice_ratio=1.0, shuffle=False):
        """
        Creates a slice of the total dataset, contianing
        'data_slice_ration * 100%' of the original data.
        Optinally, shuffle the data, then iterate over
        pair_ids and generate batches for every pair_id.
        """

        max_index = int(data_slice_ratio * len(self.unique_pair_ids))
        pair_id_slice = self.unique_pair_ids[:max_index]

        if shuffle:  # Don't forget to shuffle the training data.
            random.shuffle(pair_id_slice)

        for pair_id in pair_id_slice:
            # pair_id = self.unique_pair_ids[i % len(self.unique_pair_ids)]
            batch_indices = self.generate_batch_for_pair_id(pair_id, batch_size)
            for b in batch_indices:
                yield self.get_array_for_item(b), self.df.loc[b, "pair_id"]

    def generate_batch_for_pair_id(self, pair_id, batch_size):
        """
        Generates a batch of size 'batch_size' of positive
        and negative samples for a given pair_id. The batch
        consists of up to 50% positive samples (pair_id matches)
        and all remaining samples are negative (pair_id do not match).

        Returns list of index referencing the samples in self.df.
        """
        is_correct_pair_id_map = self.df["pair_id"] == pair_id
        positive = self.df[is_correct_pair_id_map]["image"]
        negative = self.df[~is_correct_pair_id_map]["image"]

        batch_indices = []

        if len(positive) < batch_size // 2:
            batch_indices.extend(positive.index)
        else:
            batch_indices.extend(positive.sample(batch_size // 2).index)

        remainder = batch_size - len(batch_indices)
        batch_indices.extend(negative.sample(remainder).index)
        return batch_indices

    def get_array_for_item(self, i):
        path = self.df.loc[i, "image"]
        return path

    def get_keras_dataset(self, batch_size=64, training_size_ratio=1.):
        training_size = int(training_size_ratio * len(self.df["pair_id"].unique()))
        dataset = tf.data.Dataset.from_generator(self.tf_generator, args=[batch_size, training_size],
                                                 output_types=(tf.string, tf.float16)
                                                 )
        return dataset, training_size


if __name__ == "__main__":
    train_df = joblib.load(os.path.join(get_project_dir(),
                                        "data",
                                        "processed",
                                        "category_id_1_deepfashion_train.joblib"))

    generator = GenericTripletBatchGenerator(train_df)
    for g in generator.tf_generator(16, data_slice_ratio=0.1):
        print(g)
