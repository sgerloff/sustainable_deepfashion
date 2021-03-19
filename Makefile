CATEGORY_ID = 1
MIN_PAIR_COUNT = 20
INSTRUCTION =
DEEPFASHION_DATA = "train"

setup-data: setup-deepfashion-train-data setup-deepfashion-validation-data merge-database set-deepfashion-merged preprocess
setup-deepfashion-train-data: set-deepfashion-train download extract database
setup-deepfashion-validation-data: set-deepfashion-validation download extract database

setup-gc: fetch-extract-gc set-deepfashion-train database preprocess set-deepfashion-validation database preprocess

set-deepfashion-merged:
	$(eval DEEPFASHION_DATA := merged)

set-deepfashion-train:
	$(eval DEEPFASHION_DATA := train)

set-deepfashion-validation:
	$(eval DEEPFASHION_DATA := validation)

download:
	mkdir -p data/raw
	python -m src.data.setup_data --data="$(DEEPFASHION_DATA)"

extract:
	mkdir -p data/intermediate
	unzip -n -d data/intermediate/ data/raw/$(DEEPFASHION_DATA).zip

database:
	mkdir -p data/processed
	python -m src.data.write_deepfashion2_database --input="$(shell pwd)/data/intermediate/$(DEEPFASHION_DATA)" --output="data/processed/deepfashion_$(DEEPFASHION_DATA).joblib"

merge-database:
	mkdir -p data/processed
	python -m src.data.merge_databases --inputs "$(shell pwd)/data/processed/deepfashion_train.joblib" "$(shell pwd)/data/processed/deepfashion_validation.joblib" --output "$(shell pwd)/data/processed/deepfashion_merged.joblib"

preprocess:
	python -m src.data.preprocess_data --input="data/processed/deepfashion_$(DEEPFASHION_DATA).joblib" --output="$(shell pwd)/data/processed/category_$(CATEGORY_ID)_min_count_$(MIN_PAIR_COUNT)/" --category=$(CATEGORY_ID) --min_count=$(MIN_PAIR_COUNT)

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

train:
	python -m src.train_from_instruction --instruction=$(INSTRUCTION)
