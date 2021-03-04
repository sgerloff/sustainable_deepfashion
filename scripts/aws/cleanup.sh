#!/bin/bash

echo "Save model to S3-Bucket:"
aws s3 cp $1 s3://sustainable-deepfashion/$1
echo "Zip tensorboard logs"
TF_LOG=${2}.zip
zip -r $TF_LOG $2
echo "Save logs to S3-Bucket"
aws s3 cp $TF_LOG s3://sustainable-deepfashion/$TF_LOG
rm $TF_LOG