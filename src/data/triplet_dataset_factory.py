from src.data.triplet_batch_generator import GenericTripletBatchGenerator
import tensorflow as tf

from src.utility import get_project_dir
import matplotlib.pyplot as plt
import joblib, os


class TripletDatasetFactory:
    def __init__(self, database, preprocessor=(lambda x: x), input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.triplet_generator = GenericTripletBatchGenerator(database)
        self.preprocessor = preprocessor

    def get_dataset(self, batch_size=64, data_slice_ratio=1.0, shuffle=False):
        dataset = tf.data.Dataset.from_generator(self.triplet_generator.tf_generator,
                                                 args=[batch_size, data_slice_ratio, shuffle],
                                                 output_types=(tf.string, tf.float16)
                                                 )

        dataset = dataset.map(self.preprocess, num_parallel_calls=-1)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(1)
        return dataset

    def preprocess(self, x, y):
        x = tf.io.read_file(x)
        x = tf.io.decode_jpeg(x, channels=3)
        x = tf.image.resize(x, [self.input_shape[0], self.input_shape[1]])

        x = self.augment(x)

        x = self.preprocessor(x)
        y = tf.reshape(y, ())
        return x, y

    def augment(self, x):
        return x


class AugmentedTripletDatasetFactory(TripletDatasetFactory):
    def __init__(self, database, preprocessor=(lambda x: x), input_shape=(224, 224, 3)):
        super().__init__(database, preprocessor=preprocessor, input_shape=input_shape)
        self.randomRotation = tf.keras.layers.experimental.preprocessing.RandomRotation(0.5, fill_mode="wrap")

    def augment(self, x):
        tmp_shape = x.shape
        x = self.randomRotation(tf.expand_dims(x, axis=0))
        x = tf.reshape(x, tmp_shape)

        x = tf.image.random_brightness(x, 64)
        x = tf.image.random_contrast(x, 0.7, 1.)
        x = tf.image.random_hue(x, 0.05)
        x = tf.image.random_saturation(x, 0.6, 1.0)
        x = tf.clip_by_value(x, 0., 255.)
        x = tf.cast(x, tf.float16)
        return x


if __name__ == "__main__":
    train_df = joblib.load(os.path.join(get_project_dir(),
                                        "data",
                                        "processed",
                                        "category_id_1_deepfashion_train.joblib"))

    factory = AugmentedTripletDatasetFactory(train_df, tf.keras.applications.efficientnet.preprocess_input)
    dataset = factory.get_dataset(batch_size=16, shuffle=True)

    plt.figure(figsize=(16, 16))
    for X, y in dataset.take(1):
        for i in range(X.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(tf.cast(X[i] / 255, float))
            plt.axis("off")
            plt.title(f"pair_id = {int(y[i])}")
    plt.show()
