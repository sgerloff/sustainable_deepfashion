import json, os
from src.utility import get_project_dir

import tensorflow as tf

from src.models.simple_conv2d_triplet_model_factory import SimpleConv2DTripletModelFactory

model_metadata = None
with open(os.path.join(get_project_dir(), "models", "simple_conv2d.meta")) as file:
    model_metadata = json.load(file)


model_path = os.path.join(get_project_dir(), model_metadata["saved_model"])

model = tf.keras.models.load_model(model_path, compile=False)

# print(model)
