import joblib, os
from src.utility import get_project_dir

def rebase_path(path):
    path = os.path.normpath(path)
    list_of_dir = path.split(os.sep)
    args = [get_project_dir(), "data", "intermediate"]
    args.extend(list_of_dir[-3:])
    return os.path.join(*args)


dataframe_file = os.path.join(get_project_dir(), "data", "intermediate", "own_dataframe.joblib")
df = joblib.load(dataframe_file)

df["image"] = df["image"].apply(lambda x: rebase_path(x))
df["pair_id"] = df["pair_id"].astype(int)

dataframe_output_file = os.path.join(get_project_dir(), "data", "processed", "own_dataframe.joblib")
joblib.dump(df, dataframe_output_file)
