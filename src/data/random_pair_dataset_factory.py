from src.data.flat_dataset_factory import FlatDatasetFactory
from src.instruction_utility import *

import matplotlib.pyplot as plt
import tensorflow as tf


class RandomPairDatasetFactory(FlatDatasetFactory):
    def __init__(self, database, preprocessor=(lambda x: x), input_shape=(224, 224, 3), max_pair_ids=None):
        super().__init__(database, preprocessor=preprocessor, input_shape=input_shape)
        self.max_pair_ids = max_pair_ids

    def generator(self):
        unique_pair_id = self.df["pair_id"].unique()

        if isinstance(self.max_pair_ids, int):
            unique_pair_id = unique_pair_id[:self.max_pair_ids]

        for pair_id in unique_pair_id:
            tmp_df = self.df[self.df["pair_id"] == pair_id].sample(2, replace=False)
            for i in range(len(tmp_df)):
                yield tmp_df["image"].iloc[i], tmp_df["pair_id"].iloc[i]


if __name__ == "__main__":
    train_df = load_dataframe("data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib")

    factory = RandomPairDatasetFactory(train_df)
    dataset = factory.get_dataset(batch_size=16, shuffle=False)

    plt.figure(figsize=(16, 16))
    for X, y in dataset.take(1):
        for i in range(X.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(tf.cast(X[i] / 255, float))
            plt.axis("off")
            plt.title(f"pair_id = {int(y[i])}")
    plt.show()
