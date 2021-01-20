import joblib
from models.siamese_model import get_siamese_model
import tensorflow as tf

from batch_generator import BatchGenerator

if __name__ == "__main__":

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    print(f"Load database...")
    df = joblib.load("joblib/deepfashion_train.joblib")
    print(f"Load training pairs...")
    pairs = joblib.load("joblib/pairs_training.joblib")

    batch_generator = BatchGenerator(df, pairs)

    training_size = len(pairs) // 100
    bs = 16
    dataset = tf.data.Dataset.from_generator(batch_generator.tf_generator, args=[training_size],
                                             output_types=({"input_1": tf.float16, "input_2": tf.float16}, tf.float16),
                                             output_shapes=({"input_1": [600, 600, 3], "input_2": [600, 600, 3]}, ())
                                             )

    dataset = dataset.batch(bs, drop_remainder=True).repeat()

    # df_val = joblib.load("deepfashion_validation.joblib")
    # pairs_val = joblib.load("pairs_validation.joblib")
    # val_generator = BatchGenerator(df_val, pairs_val)

    model = get_siamese_model((600, 600, 3,))
    opt = tf.keras.optimizers.Adam(1e-4)
    # The latter is needed to use full memory of the gpu (RTX 2080), which has dedicated memory for float only.
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics="accuracy")
    model.fit(dataset, steps_per_epoch=training_size // bs, epochs=10)
    model.save("test_siamese_deepfashion.h5", save_format="h5")


