import argparse
import tqdm
import src.utility as utl
import os
import requests


def verify_checksums(target_path, checksum):
    target_sha256 = utl.get_hashsum_of_file(target_path)
    if checksum == target_sha256:
        return True
    else:
        return False


def download_data(id_str, target_path):
    url = "http://d2fcl18pl6lkip.cloudfront.net/" + id_str + ".zip"
    rq = requests.get(url, stream=True)
    with open(target_path, "wb") as f:
        buffer = 64 * 1024
        pbar = tqdm.tqdm(rq.iter_content(buffer),
                         desc=id_str + ".zip",
                         unit="MB",
                         unit_scale=buffer / (1024 * 1024))
        for chunk in pbar:
            f.write(chunk)


def fetch_data(id_str):
    deepfashion_sha256 = {
        "train": "ec6f5d83f896f3abbb46bcfb9fdd6b9f544c0585344f862c214f6de899c495c7",
        "validation": "edabbdb57fae4b5039ff06e436cc0dfa15326424244bfac938e4a4d6f8db0259",
        "test": "1a85367dc9c75fbac8645e397b93af11c86bc059ab718c1eee31b559b5b4598b",
        "data-dsr": "d0072f3be3e286b4fcfa46ba670d32ba677b39c307654d804b43f9b9e1100072"
    }

    target_path = os.path.join(utl.get_project_dir(), "data", "raw", id_str + ".zip")

    if os.path.isfile(target_path):
        print("data/raw/" + id_str + ".zip already exists.")
        print("Check existing file...")
        if verify_checksums(target_path, deepfashion_sha256[id_str]):
            print("Data is verified!")
            return True
        else:
            print("Data does not match!")

    print("Download file from AWS Cloudfront...")
    download_data(id_str, target_path)

    print("Check downloaded file...")
    if verify_checksums(target_path, deepfashion_sha256[id_str]):
        print("Data is verified!")
        return True
    else:
        print("Data does not match! ERROR!!! Please check manually!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checks the existence of raw data and downloads the data if necessary.")
    parser.add_argument("--data", type=str, default="train", help="Controls which dataset is fetched train/validation/test")

    args = parser.parse_args()
    fetch_data(args.data)
