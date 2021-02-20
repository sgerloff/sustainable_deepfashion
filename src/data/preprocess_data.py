import argparse
import joblib
import os

from PIL import Image
from tqdm import tqdm


def drop_pairid(dataframe, min_count=20):
    pair_id_counts = dataframe.groupby(by="pair_id").count().min(axis=1)
    single_pair_id = pair_id_counts[pair_id_counts < min_count].index
    unique_entries = dataframe[dataframe['pair_id'].apply(lambda x: x in single_pair_id)]
    print(f"Contains {len(unique_entries)} entries with less then {min_count} pairs, which are dropped.")
    dataframe.drop(index=unique_entries.index, inplace=True)
    return dataframe

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

def clean_dataframe(df, category_id=1, min_count=20):
    # Drop items with invalid bounding box:
    df.drop(index=df[df["bounding_box"].apply(lambda x: sum(x)) == 0].index, inplace=True)
    # Select single category_id
    tmp = df[df["category_id"] == category_id].copy()
    # Drop items with insufficient pair_id pairs
    tmp = drop_pairid(tmp, min_count=min_count)
    # Crop the remaining items and rewrite the database accordingly
    return crop_and_rewrite(args.output, tmp)


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
    outfile = os.path.join(path, f"category_id_{args.category}_" + basename)
    # Write database to file
    joblib.dump(clean_dataframe(df, category_id=args.category, min_count=args.min_count), outfile)
