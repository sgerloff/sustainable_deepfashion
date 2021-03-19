import argparse
import joblib
import os

from PIL import Image
from tqdm import tqdm


def split_train_validation_pair_id(dataframe, min_count=20):
    pair_id_counts = dataframe.groupby(by="pair_id").count().min(axis=1)
    validation_pair_id = pair_id_counts[pair_id_counts < min_count].index
    train_pair_id = pair_id_counts[pair_id_counts >= min_count].index

    validation_entries = dataframe[dataframe['pair_id'].apply(lambda x: x in validation_pair_id)]
    train_entries = dataframe[dataframe['pair_id'].apply(lambda x: x in train_pair_id)]

    return dataframe.drop(index=validation_entries.index), dataframe.drop(index=train_entries.index)


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='dataframe file (joblib)')
    parser.add_argument('--output', type=str, help='target path for cropped images')
    parser.add_argument('--category', type=int, help='Item category of the deepfashion2 dataset', default=1)
    parser.add_argument('--min_count', type=int, help="Minimum number of items per pair_id", default=20)
    args = parser.parse_args()

    df = joblib.load(args.input)
    # Define output file
    path = os.path.dirname(args.input)
    basename = os.path.basename(args.input)
    train_outfile = os.path.join(path, f"category_id_{args.category}_min_pair_count_{args.min_count}_deepfashion_train")
    validation_outfile = os.path.join(path, f"category_id_{args.category}_min_pair_count_{args.min_count}_deepfashion_validation")
    # Write database to file
    train, validation = split_dataframe_train_validation(df, category_id=args.category, min_count=args.min_count)
    print("Crop and rewrite training data:")
    joblib.dump(crop_and_rewrite(
        os.path.join(args.output,"train"),
        train), train_outfile)
    print("Crop and rewrite validation data:")
    joblib.dump(crop_and_rewrite(
        os.path.join(args.output, "validation"),
        validation), validation_outfile)
