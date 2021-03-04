import argparse, json, joblib, os
from src.utility import get_project_dir

from src.models.callbacks import *
#Supported models
from src.models.efficient_net_triplet import *


def load_dataframe(path):
    return joblib.load(os.path.join(get_project_dir(), path))

def load_instruction(path):
    instruction_file = os.path.join(get_project_dir(), "instructions", path)
    with open(instruction_file, "r") as file:
        instruction_ = json.load(file)
    return instruction_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Reads instructions from "<project>/instructions/". '
                    'The script will follow the instructions to setup a '
                    'training environment, train the model, and allows '
                    'to execute bash commands after finishing.')
    parser.add_argument('--instruction', type=str,
                        default='effnet_0.0_trainable_ratio.json',
                        help='relative path to instructions from <project>/instructions/')

    args = parser.parse_args()

    instruction = load_instruction(args.instruction)
    train_df = load_dataframe(instruction["train_dataframe"])
    validation_df = load_dataframe(instruction["validation_dataframe"])

    #Initialize model
    model = globals()[instruction["model"]]()
    if instruction["load"] != "None":
        model.load(instruction["load"])

    #Load callbacks from instruction files
    callbacks = []
    for key, value in instruction["callbacks"].items():
        cb = globals()[key](**value)
        callbacks.append(cb)
    instruction["train_keywords"]["callbacks"] = callbacks

    model.set_learning_rate(instruction["learning_rate"])
    model.set_trainable_ratio(instruction["trainable_basemodel_ratio"])
    history = model.train(train_df, validation_df, **instruction["train_keywords"])

    #Execute commands to clean up, e.g. save results to persistent storage, shutdown instance and more
    script_path = os.path.join(get_project_dir(), "scripts", instruction["clean_script"])
    os.system(script_path)