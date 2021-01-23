import argparse
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to data_base joblib file')
    args = parser.parse_args()

    df = joblib.load(args.input)
    #Drop items with invalid bounding box:
    df.drop(index=df[df["bounding_box"].apply(lambda x: sum(x)) == 0].index, inplace=True)
    #Remove items with no paired entries:
    pair_id_counts = df.groupby(by="pair_id").count().min(axis=1)
    single_pair_id = pair_id_counts[pair_id_counts == 1].index
    unique_entries = df[df['pair_id'].apply(lambda x: x in single_pair_id)]
    print(f"Contains {len(unique_entries)} entries with unique pair_id, which we will drop.")
    df.drop(index=unique_entries.index, inplace=True)