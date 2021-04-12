import joblib
import os
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

def initialize_dataframe():
    df = pd.DataFrame(columns=['segmentation', 'scale', 'viewpoint', 'zoom_in',
                               'landmarks', 'style', 'bounding_box', 'category_id',
                               'occlusion', 'category_name', 'image', 'pair_id', 'source'])
    return df

def append_to_database_and_resize_images(df, subfolder, target_path, pair_id, output_shape=(224, 224)):
    path = os.path.join(get_project_dir(), 'data', 'vinted', subfolder)
    target_path = os.path.join(target_path, subfolder)
    if not os.path.isdir(target_path):
        os.makedirs(target_path)

    if not df.empty:
        index = df.index.max() + 1
    else:
        index = 0

    for item in tqdm(os.listdir(path)):
        directory = os.path.join(path, item)
        if len(os.listdir(directory)) == 0:
            continue

        item_path = os.path.join(target_path, item)
        if not os.path.isdir(item_path):
            os.makedirs(item_path)

        for filename in os.listdir(directory):
            old_filepath = os.path.join(directory, filename)
            new_filename = f'vinted_{index:05d}_{pair_id}'
            new_filepath = os.path.join(item_path, new_filename)

            image = Image.open(old_filepath)
            image_facing_up = ImageOps.exif_transpose(image)
            image_width = image_facing_up.size[0]
            image_height = image_facing_up.size[1]
            min_width = output_shape[0]
            min_height = output_shape[1]
            i = min(min_width/image_width, min_height/image_height)
            a = max(min_width/image_width, min_height/image_height)
            image_facing_up.thumbnail((output_shape[0]*a/i, output_shape[1]*a/i), Image.ANTIALIAS)
            image_facing_up.save(new_filepath, format='JPEG')

            df.loc[index, 'style'] = 1
            df.loc[index, 'bounding_box'] = [0, 0, output_shape[0], output_shape[1]]
            df.loc[index, 'category_id'] = 1
            df.loc[index, 'category_name'] = 'short sleeve top'
            df.loc[index, 'image'] = new_filepath
            df.loc[index, 'pair_id'] = pair_id
            df.loc[index, 'source'] = 'user'

            index += 1

        pair_id += 1

    return df, pair_id

def write_database(target_path, output_shape):
    path = os.path.join(get_project_dir(), 'data', 'vinted')
    df = initialize_dataframe()
    pair_id = 1

    for subfolder in os.listdir(path):
        if subfolder == '.gitignore':
            continue
        else:
            df, pair_id = append_to_database_and_resize_images(df, subfolder, target_path, pair_id, output_shape)

    return df

def get_project_dir():
    py_path = os.path.abspath(__file__)  # <project_dir>/src/data/setup_own_data.py
    data_dir = os.path.dirname(py_path)  # <project_dir>/src/data
    src_dir = os.path.dirname(data_dir)  # <project_dir>/src
    return os.path.dirname(src_dir)  # <project_dir>

if __name__ == '__main__':
    target_path = os.path.join(get_project_dir(), 'data', 'vinted-processed')
    df = write_database(target_path, output_shape=(224, 224))
    output_path = os.path.join(get_project_dir(), 'data', 'vinted-processed', 'vinted_dataframe.joblib')
    joblib.dump(df, output_path)