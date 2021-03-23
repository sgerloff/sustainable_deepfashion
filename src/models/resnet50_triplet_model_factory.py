from src.models.efficient_net_triplet_model_factory import EfficientNetB0TripletModelFactory
import tensorflow as tf


class ResNet50V2TripletModelFactory(EfficientNetB0TripletModelFactory):
    def set_basemodel(self):
        resnet50v2 = tf.keras.applications.ResNet50V2(input_shape=self.input_shape,
                                                      include_top=False,
                                                      weights="imagenet")

        resnet50v2.trainable = False
        return resnet50v2
