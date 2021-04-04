from src.instruction_utility import *
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from src.data.flat_dataset_factory import FlatDatasetFactory

def predict_epochs(instruction_parser):
    path = os.path.join(get_project_dir(), 'docs', 'assets', 'VAE_visuals', 'input', '000029_4.jpg')
    image = Image.open(path)
    image = image.resize((224, 224))
    data_render = np.asarray(image) / 255.
    data_render = np.expand_dims(data_render, axis=0)

    weight_files_list = sorted([os.path.join(get_visuals_dir(), file) for file in os.listdir(get_visuals_dir())])
    reconstruction_dict = {}

    for file in tqdm(weight_files_list):
        model = instruction_parser.get_model()
        model.compile(optimizer=instruction_parser.get_optimizer())
        model.built = True
        model.load_weights(file)
        reconstruction = model.predict(data_render)
        reconstruction_dict[file] = reconstruction

    return reconstruction_dict

def save_image_reconstructions(reconstruction_dict):
    index = 1
    for file in reconstruction_dict.keys():
        array = (reconstruction_dict[file][0] * 255).astype(np.uint8)
        image = Image.fromarray(array)
        target_path = os.path.join(get_project_dir(), 'docs', 'assets', 'VAE_visuals', 'reconstructions')
        image.save(f'{target_path}/{index:03d}.jpeg')
        index += 1

def predict_embeddings(instruction_parser):
    database = load_dataframe("data/processed/category_id_1_min_pair_count_10_deepfashion_train.joblib")
    model_factory = instruction_parser.model_factory
    factory = FlatDatasetFactory(database, preprocessor=model_factory.preprocessor())
    data = factory.get_dataset(batch_size=385, shuffle=False)

    label_batches = []
    for x, y in data.take(1):
        images_preprocessed = x
        label_batches.append(y)

    labels = np.concatenate(label_batches)


    weight_files_list = sorted([os.path.join(get_visuals_dir(), file) for file in os.listdir(get_visuals_dir())])
    embedding_dict = {}

    for file in tqdm(weight_files_list):
        model = instruction_parser.get_model()
        model.compile(optimizer=instruction_parser.get_optimizer())
        model.built = True
        model.load_weights(file)
        _, _, embeddings = model.encoder.predict(images_preprocessed)
        embedding_dict[file] = embeddings

    return embedding_dict, labels

def save_PCA_images(embedding_dict, labels):
    database = load_dataframe("data/processed/category_id_1_min_pair_count_10_deepfashion_train.joblib")
    index = 1
    for file in embedding_dict.keys():
        pca = PCA(n_components=2)
        embeddings = embedding_dict[file]
        pca.fit(embeddings)
        components = pca.transform(embeddings)
        pca_df = pd.DataFrame(components, columns=['pc1', 'pc2'])
        pca_df['pair_id'] = labels
        pca_df['pair_id'] = pca_df['pair_id'].astype(int)
        pca_df['pair_id'] = pca_df['pair_id'].astype(str)

        fig = px.scatter(pca_df,
                         x='pc1',
                         y='pc2',
                         color='pair_id',
                         range_x=(-30, 30),
                         range_y=(-30, 30),
                         color_discrete_sequence=px.colors.qualitative.Dark24,
                         width=600,
                         height=500)

        target_path = os.path.join(get_project_dir(), 'docs', 'assets', 'VAE_visuals', 'embeddings')
        fig.write_image(f'{target_path}/{index:03d}.jpeg')
        index += 1




def get_project_dir():
    file_path = os.path.abspath(__file__)  # <project_dir>/src/epoch_visualizations.py
    file_dir = os.path.dirname(file_path)  # <project_dir>/src
    return os.path.dirname(file_dir)  # <project_dir>

def get_visuals_dir():
    visuals_dir = os.path.join(get_project_dir(), 'models', 'visuals')
    return visuals_dir

if __name__ == '__main__':
    instruction_parser = InstructionParser('VAE_conv2d_input_224_embedding_512.json')
    # reconstruction_dict = predict_epochs(instruction_parser)
    # save_image_reconstructions(reconstruction_dict)
    embedding_dict, labels = predict_embeddings(instruction_parser)
    save_PCA_images(embedding_dict, labels)