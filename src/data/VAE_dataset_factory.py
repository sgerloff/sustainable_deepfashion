import tensorflow as tf


class VAEDatasetFactory:
    def __init__(self, database, preprocessor=(lambda x: x), input_shape=(224, 224, 3)):
        self.df = database
        self.input_shape = input_shape

    def get_dataset(self, batch_size=64, shuffle=False):
        files = self.df['image'].tolist()
        dataset = tf.data.Dataset.from_tensor_slices(files)

        dataset = dataset.map(self.preprocess, num_parallel_calls=-1)
        dataset = dataset.batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(10000)
        return dataset

    def preprocess(self, filename):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = img / 255.0
        return img
