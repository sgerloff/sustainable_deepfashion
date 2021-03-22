from src.data.flat_dataset_factory import FlatDatasetFactory
from src.instruction_utility import *
import numpy as np


class TopKAccuracy:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

        self.prediction = self.model.predict(self.dataset)
        self.labels = np.concatenate([y for x, y in self.dataset], axis=0)
        
        self.top_k = {}

    def get_top_k_accuracies(self, k_list=[1,5,10]):
        self.top_k = {}
        self.top_k = self.initalize_top_k_dict(k_list)

        for i in range(self.prediction.shape[0]):
            sorted_match_bool = self.get_sorted_match_bool_for_index(i)
            for k in k_list:
                self.top_k["top_"+str(k)] += int(np.any(sorted_match_bool[1:k+1]))

        return self.normalize_top_k_dict()

    def initalize_top_k_dict(self, k_list):
        self.top_k = {}
        for k in k_list:
            self.top_k["top_"+str(k)] = 0.
        return self.top_k

    def get_sorted_match_bool_for_index(self, index):
        distances = np.linalg.norm(self.prediction - self.prediction[index], axis=1)
        sorted_index = np.argsort(distances)
        match_bool = self.labels == self.labels[index]
        return match_bool[sorted_index]

    def normalize_top_k_dict(self):
        for key, value in self.top_k.items():
            self.top_k[key] = value/self.prediction.shape[0]
        return self.top_k



if __name__ == "__main__":
    metadata_file = "simple_conv2d_embedding_size_32.meta"
    metadata = load_metadata(metadata_file)
    ip = InstructionParser(metadata["instruction"], is_dict=True)
    model = load_model_from_metadata(metadata_file)

    path_to_df = metadata["instruction"]["validation_data"]["dataframe"]
    validation_dataframe = load_dataframe(path_to_df)
    factory = FlatDatasetFactory(validation_dataframe, preprocessor=ip.model_factory.preprocessor())
    dataset = factory.get_dataset()

    topk_loop = TopKAccuracy(model, dataset)
    print(topk_loop.get_top_k_accuracies(k_list=[1,5,10]))