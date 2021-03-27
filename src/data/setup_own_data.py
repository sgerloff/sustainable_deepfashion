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

def write_database_and_resize_images(target_path, output_shape=(224, 224)):
    df = initialize_dataframe()

    path = os.path.join(get_project_dir(), 'data', 'raw-data-dsr', 'pair-directories')
    index = 0

    for pair_id in tqdm(os.listdir(path)):
        directory = os.path.join(path, pair_id)
        pair_id_path = os.path.join(target_path, pair_id)
        if not os.path.isdir(pair_id_path):
            os.makedirs(pair_id_path)

        for filename in os.listdir(directory):
            old_filepath = os.path.join(directory, filename)
            new_filename = f'{index:05d}_{pair_id}'
            new_filepath = os.path.join(pair_id_path, new_filename)

            image = Image.open(old_filepath)
            image_facing_up = ImageOps.exif_transpose(image)
            new_image = image_facing_up.resize(output_shape)
            new_image.save(new_filepath, format='JPEG')

            df.loc[index, 'style'] = 1
            df.loc[index, 'bounding_box'] = [0, 0, output_shape[0], output_shape[1]]
            df.loc[index, 'category_id'] = 1
            df.loc[index, 'category_name'] = 'short sleeve top'
            df.loc[index, 'image'] = new_filepath
            df.loc[index, 'pair_id'] = pair_id
            df.loc[index, 'source'] = 'user'

            index += 1

    return df

def get_project_dir():
    py_path = os.path.abspath(__file__)  # <project_dir>/src/data/setup_own_data.py
    data_dir = os.path.dirname(py_path)  # <project_dir>/src/data
    src_dir = os.path.dirname(data_dir)  # <project_dir>/src
    return os.path.dirname(src_dir)  # <project_dir>

if __name__ == '__main__':
    target_path = os.path.join(get_project_dir(), 'data', 'data-dsr', 'pair-directories')
    df = write_database_and_resize_images(target_path, output_shape=(224, 224))
    output_path = os.path.join(get_project_dir(), 'data', 'data-dsr', 'own_dataframe.joblib')
    joblib.dump(df, output_path)




