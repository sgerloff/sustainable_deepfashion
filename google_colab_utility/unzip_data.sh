#!/bin/bash

read -s -p "Deepfashion2 Dataset Password: " pwd
unzip -n -d data/intermediate/ -P $pwd data/raw/train.zip
unzip -n -d data/intermediate/ -P $pwd data/raw/validation.zip