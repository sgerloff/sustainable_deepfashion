from src.instruction_utility import *
from src.data.random_pair_dataset_factory import RandomPairDatasetFactory
from src.metrics.top_k_from_dataset import TopKAccuracy

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute top-k-accuracies of model')
    parser.add_argument('--metafile', default="simple_conv2d_embedding_size_32.meta", type=str,
                        help='path to metadata file')

    args = parser.parse_args()

    model = load_model_from_metadata(os.path.basename(args.metafile))

    dataframe = load_dataframe("/home/sascha/Documents/Projects/fashion_one_shot_test/data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib")
    factory = RandomPairDatasetFactory(dataframe)
    dataset = factory.get_dataset(batch_size=16, shuffle=False)

    topk = TopKAccuracy(model, dataset)
    print(topk.get_top_k_accuracies(k_list=[1,5,10,15,20]))