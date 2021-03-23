import argparse
import joblib
import json
import os

import pandas as pd
from tqdm import tqdm


def read_data(path_to_data):
    tmp = []
    number_of_files = len(os.listdir(os.path.join(path_to_data, "annos")))
    ignore_count = 0
    for i in tqdm(range(1, number_of_files + 1)):
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
            else:
                ignore_count += 1
    print(f"Found {ignore_count} items with style 0, which we will ignore.")
    df = pd.DataFrame(tmp)

    #expand bounding boxes
    box_keys=[]
    for i in range(4):
        new_key = "box_" + str(i)
        df[new_key] = df["bounding_box"].apply(lambda x: x[i])
        box_keys.append(new_key)

    subset = ["pair_id", "image"]
    subset.extend(box_keys)

    duplicate_bool = df.duplicated(subset=subset)
    print(f"Found {len(df[duplicate_bool])} duplicated entries, which we will drop")
    return df[~duplicate_bool].drop(columns=box_keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to training set data')
    parser.add_argument('--output', type=str, help='output file (joblib)')
    args = parser.parse_args()

    df = read_data(args.input)
    joblib.dump(df, args.output)
