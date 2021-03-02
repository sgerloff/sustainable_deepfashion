from src.utility import get_project_dir

import joblib
import json
import os
import time

from src.models.efficient_net_triplet import EfficientNetTriplet

if __name__ == "__main__":
    effnet_triplet = EfficientNetTriplet()

    train_df = joblib.load(os.path.join(get_project_dir(),
                                        "data",
                                        "processed",
                                        "category_id_1_deepfashion_train.joblib"))

    validation_df = joblib.load(os.path.join(get_project_dir(),
                                             "data",
                                             "processed",
                                             "category_id_1_deepfashion_validation.joblib"))

    # Train with basemodel frozen:
    effnet_triplet.basemodel.trainable = False
    effnet_triplet.train(train_df, validation_df, epochs=10, training_ratio=1., batch_size=64)

    # Unfreeze some layers from the basemodel (last half):
    effnet_triplet.set_trainable_ratio(0.5)
    history = effnet_triplet.train(train_df, validation_df, epochs=100, training_ratio=1., batch_size=64)

    with open(os.path.join(get_project_dir(),
                           "reports",
                           "triplet_history_" + time.strftime("%Y%m%d") + ".json"),
              "w") as report:
        json.dump(history.history, report)
