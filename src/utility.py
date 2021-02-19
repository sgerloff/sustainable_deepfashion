import os
import hashlib


def get_project_dir():
    file_path = os.path.abspath(__file__)  # <project_dir>/src/utility.py
    file_dir = os.path.dirname(file_path)  # <project_dir>/src
    return os.path.dirname(file_dir)  # <project_dir>


def get_hashsum_of_file(path_to_file):
    buffer = 64*1024  # 64k bytes
    sha256 = hashlib.sha256()
    with open(path_to_file, "rb") as f:
        while True:
            data = f.read(buffer)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()
