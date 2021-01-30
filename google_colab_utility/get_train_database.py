import argparse
import joblib

def get_clean_database(path, min_counts=2):
    df = joblib.load(path)
    #Drop items with invalid bounding box:
    df.drop(index=df[df["bounding_box"].apply(lambda x: sum(x)) == 0].index, inplace=True)
    #Remove items with no paired entries:
    pair_id_counts = df.groupby(by="pair_id").count().min(axis=1)
    single_pair_id = pair_id_counts[pair_id_counts < min_counts].index
    unique_entries = df[df['pair_id'].apply(lambda x: x in single_pair_id)]
    print(f"Contains {len(unique_entries)} entries with less then {min_counts} pairs, which we will drop.")
    df.drop(index=unique_entries.index, inplace=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to data_base joblib file')
    args = parser.parse_args()

    df = get_clean_database(args.input)
    df.info()
