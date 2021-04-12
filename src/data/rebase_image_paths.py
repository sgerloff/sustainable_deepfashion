import argparse, joblib, os
from src.utility import get_project_dir

def rebase_path(path):
    path = os.path.normpath(path)
    list_of_dir = path.split(os.sep)
    args = [get_project_dir(), "data", "intermediate"]
    args.extend(list_of_dir[-3:])
    return os.path.join(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebase the image paths of data in data/intermediate.")
    parser.add_argument("--dataframe", type=str, default="vinted_dataframe.joblib", help="Name of the dataframe in data/intermediate")

    args = parser.parse_args()

    dataframe_file = os.path.join(get_project_dir(), "data", "intermediate", args.dataframe)
    df = joblib.load(dataframe_file)

    df["image"] = df["image"].apply(lambda x: rebase_path(x))
    df["pair_id"] = df["pair_id"].astype(int)

    dataframe_output_file = os.path.join(get_project_dir(), "data", "processed", args.dataframe)
    joblib.dump(df, dataframe_output_file)
