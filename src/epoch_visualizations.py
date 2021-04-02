from src.instruction_utility import *
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from PIL import Image

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
        model.built=True
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



def get_project_dir():
    file_path = os.path.abspath(__file__)  # <project_dir>/src/epoch_visualizations.py
    file_dir = os.path.dirname(file_path)  # <project_dir>/src
    return os.path.dirname(file_dir)  # <project_dir>

def get_visuals_dir():
    visuals_dir = os.path.join(get_project_dir(), 'models', 'visuals')
    return visuals_dir

if __name__ == '__main__':
    instruction_parser = InstructionParser('VAE_conv2d_input_224_embedding_512.json')
    reconstruction_dict = predict_epochs(instruction_parser)
    save_image_reconstructions(reconstruction_dict)