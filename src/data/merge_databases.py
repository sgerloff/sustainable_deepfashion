import pandas as pd
import argparse, joblib

def save_merge_dataframes(list_of_df):
    assert len(list_of_df) >= 2, "list of dataframes is smaller then 2, no merging possible"
    latest_max = 0
    for df in list_of_df:
        assert "pair_id" in df.keys(), "pair_id key is not in the dataframe"
        df["pair_id"] = df["pair_id"].apply(lambda x: x + latest_max)
        latest_max = max(df["pair_id"].unique())
    return pd.concat(list_of_df).reset_index()

def rewrite_pair_ids(dataframe):
    assert "pair_id" in dataframe.keys()
    pair_id_map = { pair_id: i+1 for i, pair_id in enumerate(dataframe["pair_id"].unique()) }

    dataframe["pair_id"] = dataframe["pair_id"].apply(lambda pair_id: pair_id_map[pair_id])
    return dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, nargs="+", help='List of dataframes to merge')
    parser.add_argument('--output', type=str, help='output file (joblib)')
    args = parser.parse_args()

    print("Read dataframes from joblib files...")
    list_of_df = [ joblib.load(file) for file in args.inputs ]
    print("Safely merge dataframes, retaining uniqueness of pair_ids...")
    merged_df = save_merge_dataframes(list_of_df)
    print("Rename pair_ids...")
    merged_df = rewrite_pair_ids(merged_df)
    print("Save dataframe to desired output")
    joblib.dump(merged_df, args.output)


