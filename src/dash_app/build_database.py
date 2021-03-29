from src.instruction_utility import *
from src.utility import get_project_dir

from src.data.flat_dataset_factory import FlatDatasetFactory

if __name__ == "__main__":
    metadata_file = "simple_conv2d_embedding_size_16.meta"
    model = load_model_from_metadata(metadata_file)
    metadata = load_metadata(metadata_file)

    ip = InstructionParser(metadata["instruction"], is_dict=True)

    dataframe = load_dataframe("data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib")
    dataset_factory = FlatDatasetFactory(dataframe,
                                         preprocessor=ip.model_factory.preprocessor(),
                                         input_shape=ip.model_factory.input_shape)
    dataset = dataset_factory.get_dataset(batch_size=64, shuffle=False)
    prediction = model.predict(dataset)

    output_dataframe = dataframe[["image"]].copy()
    output_dataframe["prediction"] = list(prediction)
    output_dataframe["prediction"] = output_dataframe["prediction"].apply(lambda x: list(x))

    output_dataframe.to_csv(os.path.join(get_project_dir(), "data", "processed", "dash_test_predictions.csv"))
