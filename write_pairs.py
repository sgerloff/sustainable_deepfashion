import argparse
import joblib
import numpy as np
from tqdm import tqdm

def get_positive_pairs(df, CAT_ID=1):
    cat_shop = df[((df["category_id"] == CAT_ID) & (df["source"] == "shop"))]
    cat_user = df[((df["category_id"] == CAT_ID) & (df["source"] == "user"))]

    unique_pair_id = df["pair_id"].unique()

    positive_pairs = []

    for i in tqdm(unique_pair_id):
        shop_index = list(cat_shop[cat_shop["pair_id"] == i].index)
        user_index = list(cat_user[cat_user["pair_id"] == i].index)
        if shop_index and user_index:
            for s in shop_index:
                for u in user_index:
                    positive_pairs.append((s, u, 1))

    return positive_pairs


def get_negative_pairs(df, max_number, CAT_ID=1):
    negative_pairs = []
    cat_shop = df[((df["category_id"] == CAT_ID) & (df["source"] == "shop"))]
    cat_user = df[((df["category_id"] == CAT_ID) & (df["source"] == "user"))]
    number_of_samples = min(len(cat_shop["pair_id"]), len(cat_user["pair_id"]) ) // 10
    while len(set(negative_pairs)) < max_number:
        shop = cat_shop.sample(n=number_of_samples)
        user = cat_user.sample(n=number_of_samples)

        map_pair_id_index = (np.array(shop["pair_id"].values) != np.array(user["pair_id"].values))
        z = np.zeros((len(map_pair_id_index),)).astype(int)

        tmp = list(zip(shop[map_pair_id_index].index, user[map_pair_id_index].index, z))
        negative_pairs.extend(tmp)

    negative_pairs = list(set(negative_pairs))

    assert len(negative_pairs[:max_number]) == len(set(negative_pairs[:max_number]))
    return negative_pairs[:max_number]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to training database')
    parser.add_argument('--output', type=str, help='output file (joblib)')
    args = parser.parse_args()

    print("Read database...")
    df = joblib.load(args.input)
    print("Generate positive pairs...")
    pairs = []
    pairs.extend( get_positive_pairs(df, CAT_ID=1) )
    pairs.extend( get_negative_pairs(df, len(pairs), CAT_ID=1) )
    print("Generate negative pairs...")
    np.random.shuffle(pairs)

    print(f"Write pairs to {args.output}")
    joblib.dump(pairs, args.output)


