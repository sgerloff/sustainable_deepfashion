from src.models.efficient_net_triplet_model_factory import EfficientNetB0TripletModelFactory
import tensorflow as tf


class VGG19TripletModelFactory(EfficientNetB0TripletModelFactory):
    def set_basemodel(self):
        vgg19 = tf.keras.applications.VGG19(input_shape=self.input_shape,
                                            include_top=False,
                                            weights="imagenet")

        vgg19.trainable = False
        return vgg19