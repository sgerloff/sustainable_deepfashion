import argparse
import joblib
import os

from PIL import Image
from tqdm import tqdm


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
            print(bounding, im.size)
    return df_cat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='dataframe file (joblib)')
    parser.add_argument('--output', type=str, help='target path for cropped images')
    parser.add_argument('--category', type=int, help='Item category of the deepfashion2 dataset')
    args = parser.parse_args()

    df = joblib.load(args.input)
    tmp = df[df["category_id"] == args.category].copy()
    tmp = crop_and_rewrite(args.output, tmp)
    joblib.dump(tmp, f"category_id_{args.category}_" + args.input)

