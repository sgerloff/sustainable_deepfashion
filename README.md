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

## Setup data (general)
To setup the data needed for training you can use the makefile provided.
You will be prompted to enter the [password](https://github.com/switchablenorms/DeepFashion2) required to unzip the files.

```bash
make setup-data CATEGORY_ID=1 MIN_PAIR_COUNT=20
```

This command will download the crop-traindata from AWS CloudFront, unzip the data, and finally process the data for training.
The ```CATEGORY_ID``` defines which type of items is trained, while the ```MIN_PAIR_COUNT``` defines the minimum number of items per ```pair_id```.
However, this script does not clean up the downloaded and intermediate data. If you like to save space, feel free to call ```make clean-unprocessed```.

## Setup data (Google Colab)
To save some time, we can directly take the data from google drive, which can be mounted in google colab.
To this end, you need to copy a link of the original [dataset](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing) to your own google drive.
In a google colab notebook execute:

```python
!git init
!git remote add origin https://github.com/sgerloff/sustainable_deepfashion.git
!git pull origin main

!pip install -r requirements.txt

!make setup-gc CATEGORY_ID=1 MIN_PAIR_COUNT=20
```
