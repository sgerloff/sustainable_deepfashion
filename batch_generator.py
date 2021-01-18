from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf


class BatchGenerator():
    def __init__(self, database, pairs):
        self.df = database
        self.pairs = pairs
        self.batch_size = 4
        self.offset = 0

    def get_array_for_item(self, i):

        x = tf.io.read_file(self.df.loc[i, "image"])
        x = tf.io.decode_image(x)
        bound = self.df.loc[i, "bounding_box"]
        x = tf.image.crop_to_bounding_box(x, bound[1], bound[0], bound[3] - bound[1], bound[2] - bound[0])
        x = tf.image.resize(x, [600, 600])

        return preprocess_input(x)

    def tf_generator(self, training_range):
        for i in range(training_range):
            yield {"input_1": self.get_array_for_item(self.pairs[i][0]),
                   "input_2": self.get_array_for_item(self.pairs[i][1])}, self.pairs[i][2]
