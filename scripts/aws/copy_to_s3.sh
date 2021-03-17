#!/bin/bash

echo "Save model to S3-Bucket:"

if command -v aws &> /dev/null
then
  aws s3 cp $1 s3://sustainable-deepfashion/$1
else
  shopt -s expand_aliases
  source ~/.bash_aliases
  if command -v aws &> /dev/null
  then
    aws s3 cp $1 s3://sustainable-deepfashion/$1
  else
    echo "Copy failed! The aws-cli is not installed or setup properly!"
  fi
fi