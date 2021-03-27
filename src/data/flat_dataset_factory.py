import tensorflow as tf

import matplotlib.pyplot as plt
from src.instruction_utility import load_dataframe

class FlatDatasetFactory:
    def __init__(self, database, preprocessor=(lambda x: x), input_shape=(224, 224, 3)):
        self.df = database
        self.input_shape = input_shape
        self.preprocessor = preprocessor

    def generator(self):
        for i in range(len(self.df)):
            yield self.df["image"].iloc[i], self.df["pair_id"].iloc[i]

    def get_dataset(self, batch_size=64, shuffle=False):
        dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.string, tf.float16))

        if shuffle:
            dataset = dataset.shuffle(1000)

        dataset = dataset.map(self.preprocess, num_parallel_calls=-1)
        dataset = dataset.batch(batch_size)

        return dataset

    def preprocess(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = self.preprocessor(img)
        return img, label


if __name__ == "__main__":
    train_df = load_dataframe("data/processed/category_id_1_deepfashion_train.joblib")

    factory = FlatDatasetFactory(train_df)
    dataset = factory.get_dataset(batch_size=16, shuffle=True)

    plt.figure(figsize=(16, 16))
    for X, y in dataset.take(1):
        for i in range(X.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(tf.cast(X[i] / 255, float))
            plt.axis("off")
            plt.title(f"pair_id = {int(y[i])}")
    plt.show()
