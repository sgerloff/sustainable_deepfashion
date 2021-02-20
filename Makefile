setup-data: setup-train-data setup-validation-data
setup-train-data: download-train extract-train database-train crop-train
setup-validation-data: download-validation extract-validation database-validation crop-validation

setup-gc: fetch-extract-gc database-train crop-train database-validation crop-validation

download-train:
	mkdir -p data/raw
	python -m src.data.setup_data --data="train"

extract-train:
	mkdir -p data/intermediate
	unzip -n -d data/intermediate/ data/raw/train.zip

database-train:
	mkdir -p data/processed
	python -m src.data.write_database --input="$(shell pwd)/data/intermediate/train" --output="data/processed/deepfashion_train.joblib"

crop-train:
	python -m src.data.crop_images --input="data/processed/deepfashion_train.joblib" --output="$(shell pwd)/data/processed/train/cat1/" --category="1"

download-validation:
	mkdir -p data/raw
	python -m src.data.setup_data --data="validation"

extract-validation:
	mkdir -p data/intermediate
	unzip -n -d data/intermediate/ data/raw/validation.zip

database-validation:
	mkdir -p data/processed
	python -m src.data.write_database --input="$(shell pwd)/data/intermediate/validation" --output="data/processed/deepfashion_validation.joblib"

crop-validation:
	python -m src.data.crop_images --input="data/processed/deepfashion_validation.joblib" --output="$(shell pwd)/data/processed/validation/cat1/" --category="1"

clean-unprocessed:
	rm -r data/raw data/intermediate

fetch-extract-gc:
	python -m src.google_colab_utility connect_gdrive
	ln -sfn /gdrive/MyDrive/DeepFashion2\ Dataset/*.zip data/raw/
	chmod a+x google_colab_utility/unzip_data.sh
	google_colab_utility/unzip_data.sh