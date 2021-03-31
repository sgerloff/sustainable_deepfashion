import argparse
import joblib
import os

from PIL import Image
from tqdm import tqdm

import pandas as pd
import numpy as np


def split_train_validation_pair_id(dataframe, min_count=20):
    pair_id_counts = dataframe.groupby(by="pair_id").count().max(axis=1)

    validation_pair_id = pair_id_counts[pair_id_counts < min_count].index
    train_pair_id = pair_id_counts[pair_id_counts >= min_count].index
    single_pair_id = pair_id_counts[pair_id_counts <= 1].index

    validation_entries = dataframe[dataframe['pair_id'].apply(lambda x: x in validation_pair_id)]
    train_entries = dataframe[dataframe['pair_id'].apply(lambda x: x in train_pair_id)]
    single_entries = dataframe[dataframe['pair_id'].apply(lambda x: x in single_pair_id)]

    train_df = dataframe.drop(index=validation_entries.index)
    validation_df = dataframe.drop(index=train_entries.index)
    validation_df = validation_df.drop(index=single_entries.index)

    return train_df, validation_df


def crop_and_rewrite(target_path, df_cat):
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    for i in tqdm(df_cat.index):
        # Retrieve and split image path:
        image_path = df_cat.loc[i, "image"]
        image_name = image_path.split(os.path.sep)[-1].split(".")[0]
        # Read image:
        im = Image.open(image_path)
        # Crop to bounding  box:
        bounding = df_cat.loc[i, "bounding_box"]
        imc = im.crop((bounding[0], bounding[1], bounding[2], bounding[3]))
        # Write to new destination:
        new_name = f"{image_name}_{df_cat.loc[i, 'pair_id']}.jpg"
        new_path = os.path.join(target_path, new_name)
        try:
            imc.save(new_path)
            # Update database:
            df_cat.loc[i, "image"] = new_path
        except:
            # print(bounding, im.size)
            pass
    return df_cat

def split_dataframe_train_validation(df, category_id=1, min_count=20):
    # Drop items with invalid bounding box:
    df.drop(index=df[df["bounding_box"].apply(lambda x: sum(x)) == 0].index, inplace=True)
    # Select single category_id
    df_cat1 = df[df["category_id"] == category_id].copy()
    # Drop items with insufficient pair_id pairs
    return split_train_validation_pair_id(df_cat1, min_count=min_count)

def split_off_vinted_data(dataframe):
    dataframe["isVinted"] = dataframe["image"].apply(lambda x: os.path.basename(x).startswith("vinted_"))
    vinted_index = dataframe[dataframe["isVinted"]].index
    not_vinted_index = dataframe[~dataframe["isVinted"]].index
    dataframe.drop("isVinted", inplace=True, axis=1)

    return dataframe.drop(index=vinted_index), dataframe.drop(index=not_vinted_index)


def validation_test_split(df, ratio=0.5):
    unique_pair_ids = df["pair_id"].unique()

    random_pair_id = np.random.choice(unique_pair_ids,
                                      size=int(ratio*len(unique_pair_ids)),
                                      replace=False)

    df["is_validation"] = df["pair_id"].apply(lambda x: x in random_pair_id)
    validation_index = df[df["is_validation"]].index
    test_index = df[~df["is_validation"]].index
    df.drop("is_validation", inplace=True, axis=1)
    return df.drop(index=test_index), df.drop(index=validation_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='dataframe file (joblib)', default="../../data/processed/deepfashion_merged.joblib")
    parser.add_argument('--output', type=str, help='target path for cropped images', default="../../test")
    parser.add_argument('--category', type=int, help='Item category of the deepfashion2 dataset', default=1)
    parser.add_argument('--min_count', type=int, help="Minimum number of items per pair_id", default=10)
    parser.add_argument('--min_count_vinted', type=int, help="Minimum number of items per pair_id for vinted data", default=3)
    parser.add_argument('--split_ratio', type=float, help="Ratio to split the validaiton and test set.", default=0.5)
    args = parser.parse_args()

    df = joblib.load(args.input)
    # Define output file
    path = os.path.dirname(args.input)
    basename = os.path.basename(args.input)

    train_outfile = os.path.join(path, f"category_id_{args.category}_min_pair_count_{args.min_count}_deepfashion_train.joblib")
    validation_outfile = os.path.join(path, f"category_id_{args.category}_min_pair_count_{args.min_count}_deepfashion_validation.joblib")
    test_outfile = os.path.join(path, f"category_id_{args.category}_min_pair_count_{args.min_count}_deepfashion_test.joblib")

    df, vinted_df = split_off_vinted_data(df)

    # Write database to file
    train, validation = split_dataframe_train_validation(df, category_id=args.category, min_count=args.min_count)
    train_vinted, validation_vinted = split_dataframe_train_validation(vinted_df, category_id=args.category, min_count=args.min_count_vinted)

    train = pd.concat([train, train_vinted])

    validation, test = validation_test_split(validation_vinted, ratio=args.split_ratio)



    print("Crop and rewrite training data:")
    joblib.dump(crop_and_rewrite(
        os.path.join(args.output,"train"),
        train), train_outfile)
    print("Crop and rewrite validation data:")
    joblib.dump(crop_and_rewrite(
        os.path.join(args.output, "validation"),
        validation), validation_outfile)
    print("Crop and rewrite test data:")
    joblib.dump(crop_and_rewrite(
        os.path.join(args.output, "test"),
        test), test_outfile)
