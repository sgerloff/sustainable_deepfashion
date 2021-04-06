from src.instruction_utility import *
from src.utility import get_project_dir

from src.data.flat_dataset_factory import FlatDatasetFactory

import argparse

from sklearn.decomposition import PCA
import tensorflow as tf

def write_prediction_dataframe_from_metafile(metafile,
                                             dataframe="data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib",
                                             isVAE=False):
    model = load_model_from_metadata(metafile, best_model_key="best_top_1_model")
    if isVAE:
        model = model.encoder

    metadata = load_metadata(metafile)

    ip = InstructionParser(metadata["instruction"], is_dict=True)

    dataframe = load_dataframe(dataframe)
    dataset_factory = FlatDatasetFactory(dataframe,
                                         preprocessor=ip.model_factory.preprocessor(),
                                         input_shape=ip.model_factory.input_shape)
    dataset = dataset_factory.get_dataset(batch_size=64, shuffle=False)

    if isVAE:
        _, _, prediction = model.predict(dataset)
    else:
        prediction = model.predict(dataset)

    output_dataframe = dataframe[["image"]].copy()
    output_dataframe["web_image"] = output_dataframe["image"].apply(lambda img: get_webadress_of_image(img))

    output_dataframe["prediction"] = list(prediction)
    output_dataframe["prediction"] = output_dataframe["prediction"].apply(lambda x: list(x))

    return output_dataframe, prediction, model


def get_webadress_of_image(img):
    img = remove_project_dir(img)
    list_of_dir = img.split(os.sep)
    return "http://d2fcl18pl6lkip.cloudfront.net/"+os.path.join(*list_of_dir[2:])


def get_PCA_components(prediction):
    pca = PCA(n_components=prediction.shape[1])
    pca.fit(prediction)
    return pca.components_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta',
                        type=str,
                        default="simple_conv2d_embedding_size_20_angular_d_augmented.meta",
                        help='metadata file in models/')
    parser.add_argument('--dataframe',
                        type=str,
                        default="data/processed/app_database.joblib",
                        help='dataframe file (joblib)')
    parser.add_argument('--VAE', type=bool, default=False, help="Predict using the VAE procedure")
    args = parser.parse_args()

    metadata_file = args.meta
    output_dataframe, prediction, model = write_prediction_dataframe_from_metafile(metadata_file,
                                                                                   dataframe=args.dataframe,
                                                                                   isVAE=args.VAE)


    basename = os.path.splitext(metadata_file)[0]

    # Create Folder if needed:
    output_folder = os.path.join(get_project_dir(), "data", "processed", f"{basename}")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Write dataframe to csv:
    output_dataframe.to_csv(os.path.join(output_folder, f"{basename}_predictions.csv"))

    # Write principle component to pickle
    pca_components = get_PCA_components(prediction)
    joblib.dump(pca_components, os.path.join(output_folder, f"{basename}_pca.joblib"))

    # Save trained model:
    tf.saved_model.save(model, os.path.join(output_folder, f"{basename}_saved_model"))
