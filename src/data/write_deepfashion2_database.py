import argparse
import joblib
import json
import os

import pandas as pd
from tqdm import tqdm

def read_data(path_to_data):
    tmp = []
    number_of_files = len(os.listdir(os.path.join(path_to_data, "annos")))
    for i in tqdm(range(1,number_of_files+1)):
        f = open(os.path.join(path_to_data, "annos", f"{i:06d}.json"), "r")
        data = json.load(f)
        items = [key for key in data.keys() if key.startswith("item")]
        for item in items:
            data[item]["image"] = os.path.join(path_to_data, "image", f"{i:06d}.jpg")
            data[item]["pair_id"] = data["pair_id"]
            data[item]["source"] = data["source"]
            # An item with style 0 is essentially unidentified with respect to the pair_id. Thus we drop them:
            if data[item]["style"] != 0:
                tmp.append(data[item])
    return pd.DataFrame(tmp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to training set data')
    parser.add_argument('--output', type=str, help='output file (joblib)')
    args = parser.parse_args()

    df = read_data(args.input)
    joblib.dump(df, args.output)
