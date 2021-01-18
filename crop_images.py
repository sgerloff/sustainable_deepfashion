from PIL import Image
import joblib
import os

from tqdm import tqdm

df = joblib.load("deepfashion_train.joblib")

for i in tqdm(df[df["category_id"]==1].index):
    image_path = df.loc[i, "image"]
    image_name = image_path.split("\\")[-1].split(".")[0]
    im = Image.open(image_path)
    bounding = df.loc[i, "bounding_box"]
    imc = im.crop((bounding[0], bounding[1], bounding[2], bounding[3])).resize((600, 600))
    imc.save(os.path.join(r"C:\deepfashion_data\train\image", f"{image_name}.jpg"))