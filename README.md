# Sustainable Deepfashion
Circular fashion solves 21st century developments in the fashion industry - i.e. a growing neglect about social and environmental issues surrounding the production of clothing - by trying to build a more sustainable business model. In a circular fashion economy every piece of clothing gets recycled, rented, repaired, redesigned or resold. 
At the moment it is nonetheless quite hard to find environment-friendly sources to buy or rent from. It could also be difficult to find clothing that matches your style, as a fast fashion based brand has a larger collection to choose from.

To this end, our project will focus on building an app that lets you take or upload a picture from any piece of clothing and matches this picture to similar pieces of clothing from sustainable sources, be it fashion brands or second-hand offerings. 

The strategy is to implement one-shot learning techniques developed for face-recognition and train the model to match pictures of fashion items to that of sustainable sources. Using Tensorflow, we employ transfer learning to start with a pretrained convolutional network, e.g. the EfficientNetB0. This model is trained on the deepfashion2 dataset, as well as our own data. The output of the trained model is an embedding vector, whose L2-distance to another embedding is small if the pictures contain the same (or similar) items and large otherwise.

Finally, the user provided image is checked against a database containing sustainable fashion to provide the Top 5 closest alternatives from Ebay Kleinanzeigen, sustainable brands, or similar.

# Setup
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
pip install -r requirements.txt
make setup-data CATEGORY_ID=1 MIN_PAIR_COUNT=20
```

This command will download the data from AWS CloudFront, unzip the data, and finally process the data for training.
The ```CATEGORY_ID``` defines on which type of items the network will be trained, while the ```MIN_PAIR_COUNT``` defines the minimum number of items per ```pair_id```.
However, this script does not clean up the downloaded and intermediate data. If you like to save space, feel free to call ```make clean-unprocessed``` after running the above command.

## Setup data (Google Colab)
To save some time, we can directly take the data from google drive, which can be mounted in google colab.
To this end, you first need to add a shortcut of the original [dataset](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok?usp=sharing) to your own google drive.
Afterwards you can execute the following code in a google colab notebook:

```python
!git init
!git remote add origin https://github.com/sgerloff/sustainable_deepfashion.git
!git pull origin main

!pip install -r requirements.txt

!make setup-gc CATEGORY_ID=1 MIN_PAIR_COUNT=20
```

This will clone the repository to the base folder of your colab notebook, install the requirments, and continue to setup the data. 
You will need to provide an access token to mount your google drive as well as the password for the dataset.
The entire process will take about 30 mins.

### Save the processed data to Google Drive

Chances are you dont want to repeat this setup process.
In order to save the processed data execute:

```bash
!make save-preprocessed-gc CATEGORY_ID=1
```

### Setup from already preprocessed data

And next time you start a colab notebook, you can simply load:

```python
!git init
!git remote add origin https://github.com/sgerloff/sustainable_deepfashion.git
!git pull origin main

!pip install -r requirements.txt

!make setup-preprocessed-gc CATEGORY_ID=1 MIN_PAIR_COUNT=20
```

This does speedup the setup process a lot! However, you are stuck with the preprocessed data only, until you load the rest manually.

## Manual download of the datasets
If the setup scripts fail, you may need to download the dataset from Google Drive manually. Please make sure to put them into the proper folder (```data/raw```).
Go to https://github.com/switchablenorms/DeepFashion2 and download the (train.zip, validation.zip, test.zip) dataset from their [Google Drive](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok).

