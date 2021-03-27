import tensorflow as tf


class VAEDatasetFactory:
    def __init__(self, database, preprocessor=(lambda x: x), input_shape=(224, 224, 3)):
        self.df = database
        self.input_shape = input_shape
        self.preprocessor = preprocessor

    def get_dataset(self, batch_size=64, shuffle=False):
        files = self.df['image'].tolist()
        dataset = tf.data.Dataset.from_tensor_slices(files)

        if shuffle:
            dataset = dataset.shuffle(1000)

        dataset = dataset.map(self.preprocess)
        dataset = dataset.batch(batch_size)

        return dataset

    def preprocess(self, filename):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.input_shape[0], self.input_shape[1]])
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = self.preprocessor(img)
        return img
