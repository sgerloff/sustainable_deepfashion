CATEGORY_ID = 1
MIN_PAIR_COUNT = 20
INSTRUCTION =

setup-data: setup-train-data setup-validation-data
setup-train-data: download-train extract-train database-train preprocess-train
setup-validation-data: download-validation extract-validation database-validation preprocess-validation

setup-gc: fetch-extract-gc database-train preprocess-train database-validation preprocess-validation

download-train:
	mkdir -p data/raw
	python -m src.data.setup_data --data="train"

extract-train:
	mkdir -p data/intermediate
	unzip -n -d data/intermediate/ data/raw/train.zip

database-train:
	mkdir -p data/processed
	python -m src.data.write_database --input="$(shell pwd)/data/intermediate/train" --output="data/processed/deepfashion_train.joblib"

preprocess-train:
	python -m src.data.preprocess_data --input="data/processed/deepfashion_train.joblib" --output="$(shell pwd)/data/processed/train/cat1/" --category=$(CATEGORY_ID) --min_count=$(MIN_PAIR_COUNT)

download-validation:
	mkdir -p data/raw
	python -m src.data.setup_data --data="validation"

extract-validation:
	mkdir -p data/intermediate
	unzip -n -d data/intermediate/ data/raw/validation.zip

database-validation:
	mkdir -p data/processed
	python -m src.data.write_database --input="$(shell pwd)/data/intermediate/validation" --output="data/processed/deepfashion_validation.joblib"

preprocess-validation:
	python -m src.data.preprocess_data --input="data/processed/deepfashion_validation.joblib" --output="$(shell pwd)/data/processed/validation/cat1/" --category=$(CATEGORY_ID) --min_count=$(MIN_PAIR_COUNT)

clean-unprocessed:
	rm -r data/raw data/intermediate

fetch-extract-gc:
	python -m src.google_colab_utility connect_gdrive
	ln -sfn /gdrive/MyDrive/DeepFashion2\ Dataset/*.zip data/raw/
	chmod a+x scripts/google_colab_utility/unzip_data.sh
	scripts/google_colab_utility/unzip_data.sh

save-preprocessed-gc:
	mkdir -p /gdrive/MyDrive/deepfashion_gc_save
	zip -r /gdrive/MyDrive/deepfashion_gc_save/preprocessed_cat_$(CATEGORY_ID).zip data/processed/train/cat$(CATEGORY_ID) data/processed/validation/cat$(CATEGORY_ID) data/processed/category_id_$(CATEGORY_ID)_deepfashion_train.joblib data/processed/category_id_$(CATEGORY_ID)_deepfashion_validation.joblib

setup-preprocessed-gc:
	python -m src.google_colab_utility connect_gdrive
	unzip /gdrive/MyDrive/deepfashion_gc_save/preprocessed_cat_$(CATEGORY_ID).zip

train-aws-stop:
	python -m src.train_from_instruction --instruction=$(INSTRUCTION)
	sudo shutdown now -h
