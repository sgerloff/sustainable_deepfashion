# Setup AWS instance

To avoid hassle, it seems to be best to use one of the "Deep Learning" instance settings provided. 
I have chosen the Amazon-Linux v40.
After starting and connecting to the machine chose the ```source activate tensorflow2_latest_p37``` preset.
You should create a new directory on a disk with sufficient space and navigate to this directory.

```bash
git clone https://github.com/sgerloff/sustainable_deepfashion.git
cd sustainable_deepfashion
pip install -r requirments.txt

aws s3 cp s3://sustainable-deepfashion/train.zip data/raw/
aws s3 cp s3://sustainable-deepfashion/validation.zip data/raw/
make setup-data
zip -r preprocessed_cat_1.zip data/processed/train/cat1/ data/processed/validation/cat1/ data/processed/category_id_1_deepfashion_train.joblib data/processed/category_id_1_deepfashion_validation.joblib
aws s3 cp preprocessed_cat_1.zip s3://sustainable-deepfashion/
rm preprocessed_cat_1.zip
```

Notes:

Seems like we need tensorflow 2.3.0 and need to install tensorflow_addons explicitly.