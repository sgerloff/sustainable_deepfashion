import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from src.models.efficient_net_triplet import EfficientNetTriplet
from src.utility import get_project_dir

import joblib, os, tempfile


class LearningRateEstimator:
    def __init__(self, stop_factor=4, beta=0.98):
        self.model = None
        self.stop_factor = stop_factor
        self.beta = beta

        # Define callback:
        self.record_smooth_loss_callback = tf.keras.callbacks.LambdaCallback(
            on_batch_end=lambda batch, logs: self.record_smooth_loss(batch, logs)
        )

        self.test = []
        # To be defined in reset
        self.learning_rate_tape = None
        self.loss_tape = None
        self.learning_rate_multiplier = None
        self.avg_loss = None
        self.best_loss = None
        self.stop_loss = None
        self.batch_count = None
        self.tmpWeightsFile = None

        self.reset()

    def reset(self):
        self.learning_rate_tape = []
        self.loss_tape = []
        self.learning_rate_multiplier = 1
        self.avg_loss = 0.0
        self.best_loss = 1e100
        self.stop_loss = None
        self.batch_count = 0
        self.tmpWeightsFile = None

    def record_smooth_loss(self, batch, history):
        self.test.append(batch)
        learning_rate = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rate_tape.append(learning_rate)

        loss = history["loss"]
        self.batch_count += 1
        smooth = self.smooth_avg_loss(loss)
        self.loss_tape.append(smooth)

        if self.stop_loss_explosion(smooth):
            return None

        tf.keras.backend.set_value(self.model.optimizer.lr,
                                   learning_rate * self.learning_rate_multiplier)
        return None

    def smooth_avg_loss(self, loss):
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        return self.avg_loss / (1 - self.beta ** self.batch_count)

    def stop_loss_explosion(self, smooth):
        stop_loss = self.stop_factor * self.best_loss
        if self.batch_count > 1 and smooth > stop_loss:
            self.model.stop_training = True
            return True
        if self.batch_count == 0 or smooth < self.best_loss:
            self.best_loss = smooth
        return False

    def find(self, model, dataset,
             start_learning_rate=1e-10,
             stop_learning_rate=1e1,
             epochs=None,
             steps_per_epoch=None,
             sample_size=2048,
             verbose=1):
        # Initialize all values
        self.reset()
        self.model = model
        # Set sufficient amount of epochs
        if epochs is None:
            epochs = int(np.ceil(sample_size / float(steps_per_epoch)))
        number_of_steps = epochs * steps_per_epoch

        # Define learning rate multiplier, such that start_learning_rate * learning_rate_multiplyer ** number_of_steps = stop_learning_rate
        self.learning_rate_multiplier = (stop_learning_rate / start_learning_rate) ** (1.0 / number_of_steps)

        # Save old weights
        original_learning_rate = self.save_old_state()
        tf.keras.backend.set_value(self.model.optimizer.lr, start_learning_rate)

        # Run the experiment and record loss via callback
        self.model.fit(
            dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[self.record_smooth_loss_callback],
            verbose=verbose
        )

        self.restore_old_state(original_learning_rate)

    def save_old_state(self):
        self.tmpWeightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.tmpWeightsFile)

        return tf.keras.backend.get_value(self.model.optimizer.lr)

    def restore_old_state(self, original_learning_rate):
        self.model.load_weights(self.tmpWeightsFile)
        tf.keras.backend.set_value(self.model.optimizer.lr, original_learning_rate)

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.learning_rate_tape[skipBegin:-skipEnd]
        losses = self.loss_tape[skipBegin:-skipEnd]
        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)
        plt.show()


if __name__ == "__main__":
    path_to_train_df = os.path.join(
        get_project_dir(),
        "data",
        "processed",
        "category_id_1_deepfashion_train.joblib"
    )
    train_df = joblib.load(path_to_train_df)

    effnet = EfficientNetTriplet()
    dataset, train_size = effnet.get_dataset(train_df, training_ratio=1., batch_size=32)

    learningRateEstimator = LearningRateEstimator()
    learningRateEstimator.find(effnet.model,
                               dataset,
                               start_learning_rate=1e-10,
                               stop_learning_rate=1e2,
                               steps_per_epoch=train_size)
    learningRateEstimator.plot_loss(title="Frozen basemodel EfficientNetB0")