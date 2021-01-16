import numpy as np

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


class BatchGenerator():
    def __init__(self, database, pairs):
        self.df = database
        self.pairs = pairs
        self.batch_size = 4
        self.offset = 0

    def get_array_for_item(self, i):
        im = Image.open(self.df.loc[i, "image"])
        bounding = self.df.loc[i, "bounding_box"]
        imc = im.crop((bounding[0], bounding[1], bounding[2], bounding[3])).resize((600, 600))
        x = image.img_to_array(imc)
        x = np.expand_dims(x, axis=0)

        return preprocess_input(x)

    def get_batch(self):
        x_pairs = [np.empty((self.batch_size, 600, 600, 3)) for i in range(2)]
        targets = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):
            index = self.offset + i
            x_pairs[0][i] = self.get_array_for_item(self.pairs[index][0])
            x_pairs[1][i] = self.get_array_for_item(self.pairs[index][1])
            targets[i] = self.pairs[index][2]

        return x_pairs, targets

    def generate(self, batch_size=4, train_range=16):
        self.batch_size = batch_size
        while True:
            for i in range(train_range):
                self.offset = i * self.batch_size
                p, t = self.get_batch()
                yield p, t
