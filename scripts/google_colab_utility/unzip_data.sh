#!/bin/bash

read -s -p "Deepfashion2 Dataset Password: " pwd
unzip -n -d data/intermediate/ -P $pwd data/raw/$1.zip