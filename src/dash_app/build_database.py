from src.instruction_utility import *
from src.utility import get_project_dir

from src.data.flat_dataset_factory import FlatDatasetFactory

import argparse


def write_prediction_dataframe_from_metafile(metafile, dataframe="data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib"):
    model = load_model_from_metadata(metafile, best_model_key="best_top_1_model")
    metadata = load_metadata(metafile)

    ip = InstructionParser(metadata["instruction"], is_dict=True)

    dataframe = load_dataframe(dataframe)
    dataset_factory = FlatDatasetFactory(dataframe,
                                         preprocessor=ip.model_factory.preprocessor(),
                                         input_shape=ip.model_factory.input_shape)
    dataset = dataset_factory.get_dataset(batch_size=64, shuffle=False)
    prediction = model.predict(dataset)

    output_dataframe = dataframe[["image"]].copy()
    output_dataframe["prediction"] = list(prediction)
    output_dataframe["prediction"] = output_dataframe["prediction"].apply(lambda x: list(x))

    return output_dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta',
                        type=str,
                        default="simple_conv2d_embedding_size_32-1.meta",
                        help='metadata file in models/')
    parser.add_argument('--dataframe',
                        type=str,
                        default="data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib",
                        help='dataframe file (joblib)')
    args = parser.parse_args()

    metadata_file = args.meta
    output_dataframe = write_prediction_dataframe_from_metafile(metadata_file, dataframe=args.dataframe )

    basename = os.path.splitext(metadata_file)[0]
    output_dataframe.to_csv(os.path.join(get_project_dir(), "data", "processed", f"{basename}_predictions.csv"))
