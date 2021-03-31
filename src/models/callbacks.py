import tensorflow as tf
import os

from src.data.random_pair_dataset_factory import RandomPairDatasetFactory
from src.instruction_utility import *

from src.metrics.top_k_from_dataset import TopKAccuracy, VAETopKAccuracy

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import numpy as np

def Checkpoint(path="default", file_name=True, kwargs={}):
    if file_name:
        path = os.path.join(get_project_dir(), "models", path + ".h5")
    return tf.keras.callbacks.ModelCheckpoint(path, **kwargs, save_weights_only=True)

def Tensorboard(log_dir="default", dir_name=True, kwargs={}):
    if dir_name:
        log_dir = os.path.join(get_project_dir(), "reports", "tensorboard_logs", log_dir)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, **kwargs)

#From https://github.com/bckenstler/CLR/blob/master/clr_callback.py
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class TopKValidation(tf.keras.callbacks.Callback):
    def __init__(self, dataframe="data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib",
                 epoch_frequency=10,
                 best_model_filepath=None,
                 k_list=[1,5,10],
                 preprocessor=(lambda x:x)):
        super().__init__()
        self.dataset = self.get_dataset(dataframe, preprocessor)

        self.best_model_filepath = os.path.join(get_project_dir(), "models", best_model_filepath + "_best_top_1.h5")
        self.k_list = k_list
        #Ensure that we track top-1
        if 1 not in self.k_list:
            self.k_list.append(1)

        self.distance_metric = "L2"
        self.input_shape = (224, 224, 3)

        self.epoch_frequency = epoch_frequency
        self.best_top_k = 0.

        self.top_k_log = {}

    def get_dataset(self, dataframe_path, preprocessor):
        validation_df = load_dataframe(dataframe_path)
        factory = RandomPairDatasetFactory(validation_df, preprocessor=preprocessor)
        return factory.get_dataset(batch_size=16, shuffle=False)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_frequency == 0:
            top_k_accuracies = self.get_top_k_accuracies()
            self.print_info(top_k_accuracies)
            if self.best_model_filepath is not None:
                self.save_best_weights(top_k_accuracies)

            self.top_k_log[epoch] = top_k_accuracies

    def get_top_k_accuracies(self):
        if hasattr(self.model.loss, "_fn_kwargs"):
            self.distance_metric = self.model.loss._fn_kwargs["distance_metric"]

        topk = TopKAccuracy(self.model, self.dataset, distance_metric=self.distance_metric)
        return topk.get_top_k_accuracies(k_list=self.k_list)

    def print_info(self, top_k_accuracies):
        info = f" validation [metric={self.distance_metric}]:"
        for key, value in top_k_accuracies.items():
            info = info + f"{key} = {value:0.4f}; "
        print(info)

    def save_best_weights(self, top_k_accuracies):
        if top_k_accuracies["top_1"] >= self.best_top_k:
            self.best_top_k = top_k_accuracies["top_1"]
            self.model.save_weights(self.best_model_filepath)

    def get_log(self):
        return self.top_k_log

class VAETopKValidation(TopKValidation):
    def get_top_k_accuracies(self):
        topk = VAETopKAccuracy(self.model.encoder, self.dataset)
        return topk.get_top_k_accuracies(k_list=self.k_list)

    def get_dataset(self, dataframe_path, preprocessor):
        validation_df = load_dataframe(dataframe_path)
        if hasattr(self.model, "encoder"):
            self.input_shape = (self.model.encoder.layers[0].input.shape[1],
                                self.model.encoder.layers[0].input.shape[2],
                                self.model.encoder.layers[0].input.shape[3])

        factory = RandomPairDatasetFactory(validation_df, preprocessor=preprocessor, input_shape=self.input_shape)
        return factory.get_dataset(batch_size=16, shuffle=False)


if __name__ == "__main__":
    ip = InstructionParser("simple_conv2d.json")
    model = ip.get_model()

    train_dataset = ip.get_train_dataset()
    model.compile(
        loss=ip.get_loss(),
        optimizer=ip.get_optimizer()
    )

    top_k_callback = TopKValidation(dataframe="data/processed/category_id_1_min_pair_count_10_deepfashion_validation.joblib",
                                    epoch_frequency=1,
                                    best_model_filepath="test_top_k_callback"
                                    )

    model.fit(train_dataset,
              epochs=10,
              steps_per_epoch=2,
              callbacks=[top_k_callback]
              )

    print(top_k_callback.get_log())