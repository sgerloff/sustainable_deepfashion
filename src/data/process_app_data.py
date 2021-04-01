import argparse, joblib, os
import numpy as np
from src.utility import get_project_dir, remove_project_dir

from PIL import Image


def get_random_user_images_from_dataframe(dataframe, category=1):

    user_data = dataframe[dataframe["source"] == "user"]

    user_data_cat1 = user_data[user_data["category_id"] == category]

    index_series = user_data_cat1.groupby("pair_id").apply(
        lambda frame: np.random.choice(frame.index, size=1, replace=False)
    )
    index_series = index_series.apply(lambda x: x[0])

    return user_data_cat1.loc[index_series]


def resize_and_move_image(image_path, target_path, size=300.):
    image = Image.open(image_path)
    width, height = image.size
    if width > height:
        image = image.resize((int(size), int(size*height/width)), Image.ANTIALIAS)
    else:
        image = image.resize((int(size*width/height), int(size)), Image.ANTIALIAS)

    #Get output path:
    relative_image_path = remove_project_dir(image_path)
    list_of_dir = relative_image_path.split(os.sep)
    relative_output_path = os.path.join(*list_of_dir[2:])
    output_path = os.path.join(target_path, relative_output_path)
    #Create directories if needed:
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    image.save(output_path, "JPEG")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebase the image paths of data in data/intermediate.")
    parser.add_argument("--dataframe", type=str, default="deepfashion_merged.joblib", help="Name of the dataframe in data/processed")
    parser.add_argument("--size", type=int, default=300, help="Final longest side of the image")
    parser.add_argument("--output", type=str, default="app_database.joblib", help="Output name in data/processed")
    args = parser.parse_args()


    dataframe = joblib.load(os.path.join(get_project_dir(), "data", "processed", args.dataframe))
    random_user_data = get_random_user_images_from_dataframe(dataframe, category=1)

    target_path = os.path.join(get_project_dir(), "data", "processed", os.path.splitext(args.output)[0])
    random_user_data["image"] = random_user_data["image"].apply(lambda img: resize_and_move_image(img, target_path, size=args.size))

    joblib.dump(random_user_data, os.path.join(get_project_dir(), "data", "processed", args.output))
