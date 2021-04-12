import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.decomposition import PCA
from src.instruction_utility import *

from src.data.flat_dataset_factory import FlatDatasetFactory

import os, tqdm


def get_top_k_populated_pair_ids(dataframe, k=10):
    most_populated_pair_ids = dataframe.groupby(by="pair_id").count().max(axis=1)
    most_populated_pair_ids = most_populated_pair_ids.sort_values(ascending=False)
    return list(most_populated_pair_ids[:k].index)


def get_pca_from_prediction(prediction):
    pca = PCA()
    pca.fit(prediction)
    return np.matmul(prediction, np.transpose(pca.components_[:2]))


def get_epoch_dataframe(dataframe, predictions, epoch=0, k=10):
    top_k_pair_ids = get_top_k_populated_pair_ids(dataframe, k=k)
    input_df = dataframe.copy()
    input_df["prediction"] = list(get_pca_from_prediction(predictions))
    input_df["epoch"] = epoch
    input_df["isPopulated"] = input_df["pair_id"].apply(lambda x: x in top_k_pair_ids)
    output_df = input_df[input_df["isPopulated"]]
    output_df["pred_pca1"] = output_df["prediction"].apply(lambda x: x[0])
    output_df["pred_pca2"] = output_df["prediction"].apply(lambda x: x[1])
    output_df["pair_id"] = output_df["pair_id"].astype("string")
    return output_df[["pair_id", "pred_pca1", "pred_pca2", "epoch"]]


if __name__ == "__main__":

    instruction = "simple_conv2d_embedding_size_20_angular_d_augmented.json"

    weight_dict = {
        0: None,
    }
    for i in range(1,39):
        weight_dict[i] = f"simple_conv2d_embedding_size_20_angular_d_augmented_{i:02d}.h5"

    ip = InstructionParser(instruction)
    model = ip.get_model()

    dataframe = load_dataframe(ip.instruction["train_data"]["dataframe"])
    data_factory = FlatDatasetFactory(dataframe, preprocessor=ip.model_factory.preprocessor())
    dataset = data_factory.get_dataset(shuffle=False)

    epoch_dataframes = []
    for epoch, weights in tqdm.tqdm(weight_dict.items()):
        if weights is not None:
            model.load_weights(weights)
        prediction = model.predict(dataset)
        epoch_dataframes.append(get_epoch_dataframe(dataframe, prediction, epoch=epoch, k=10))

    df = pd.concat(epoch_dataframes)

    joblib.dump(df, os.path.join(get_project_dir(),"reports", "simple_conv2d_embedding_size_20_angular_d_augmented_df.joblib"))

    fig = px.scatter(df,
                     x="pred_pca1",
                     y="pred_pca2",
                     animation_frame="epoch",
                     color="pair_id",
                     range_x=[-1, 1],
                     range_y=[-1, 1],
                     labels={
                         "pred_pca1": "x",
                         "pred_pca2": "y"
                     }
                     )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )
    fig.write_html("/home/sascha/px_test1.html")
