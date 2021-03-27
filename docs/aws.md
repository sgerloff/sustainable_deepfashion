# Training on AWS

While training with the free contingent is basically impossible, one can apply for a fund of about 300$. Given this budget, all you need to prepare is to apply for a vCPU-limit on instances that provide GPU's. We have taken p2-instances and prefer the "on-demand" type.

## Setup Data Storage

It is convenient to have all data on a mounted EBS device. 
Just create a small one, 30 GB should suffice, and note or double check the region this EBS device is created in. 
When creating the instance these regions have to match!
The training data is distributed on S3 buckets and to access them you need to configure IAM-roles and more. 
If you don't do this, you will need to use the normal make-commands to setup the data.

## Setup Instance

### Create Instance
- Create a EC2-Instance and choose ``Deep Learning AMI (Amazon Linux) Version <latest>``. 
This image comes preinstalled with lots of different environments and does cut down the setup steps significantly. 

- Choose the instance that you want to train on, such as ```p2.xlarge``
- In the configure instance tab, choose the subnet to match the EBS-device that you have created beforehand.
- Optionally, set IAM-role and whatever more you desire to change.

Next we need to hook up the EBS-device to the instance. 
Navigate to the EBS console, choose your EBS-device and attach it to the instance that you have started. 
In case you cant find your instance id, you may have ended up in different subnets.

### Mount EBS-Device
After logging in, we need to mount the EBS-device that you have created:

```bash
sudo mkdir /data
sudo mount /dev/xvdf /data
```

Your data stored on the EBS-device can now be accessed from ``/data``

### Activate Conda Env
Since cloning the environment takes about as much time as configuring a new instance, simply start the desired environment:

```bash
source activate tensorflow2_latest_p37
```

### Clone Repository
All our scripts are stored in the repository. Simply clone it to the mounted disk:

```bash
cd /data
git clone https://github.com/sgerloff/sustainable_deepfashion.git
cd sustainable_deepfashion
```

### Install dependencies

Most packages are already preinstalled. But you still need to install:

```bash
pip install tensorflow-addons
```

### [Optional] Get data and save to S3-Buckets
To save some traffic on the S3-Buckets its better to directly copy the data. However this requires setting up the permissions beforehand:

```bash
aws s3 cp s3://sustainable-deepfashion/train.zip data/raw/
aws s3 cp s3://sustainable-deepfashion/validation.zip data/raw/
make setup-data
zip -r preprocessed_cat_1.zip data/processed/train/cat1/ data/processed/validation/cat1/ data/processed/category_id_1_deepfashion_train.joblib data/processed/category_id_1_deepfashion_validation.joblib
aws s3 cp preprocessed_cat_1.zip s3://sustainable-deepfashion/
rm preprocessed_cat_1.zip
```

Next you start from scratch, simply download the preprocessed data instead and unzip from the projects base folder.
