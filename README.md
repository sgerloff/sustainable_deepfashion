# fashion_one_shot_test
This is a small test project to explore one-shot training techniques for recognizing cloths from the deepfashion data set.

# Setup
## Download the datasets
Go to https://github.com/switchablenorms/DeepFashion2 and download the (train.zip) dataset from their google drive: 

https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok

## Setup a conda environment (OPTIONAL)
It is advised to create a python environment of your choice. For example:

```bash
conda create -n fashion python=3.8
conda activate fashion
```

## Write the database from the dataset
To combine the meta data for the images to one database, you can excute the following script.
```bash
python write_database.py --input="dir/of/data" --output="deepfashion_train.joblib"
```
This script reads all the data, creates a pandas dataframe and saves it as a joblib object.

### Load the database
If you want to load the database, simply execute 

```python
import joblib
df = joblib.load("deepfashion_train.joblib")
```

### Crop images
To save space and load times for training, it is advisable to crop the images to their bounding boxes. To do so, execute the following script passing the category_id you are interested in:

```bash
python crop_images.py --input="deepfashion_train.joblib" --output="path/to/cropped_images_dir" --category="1"
```