from src.models.efficient_net_triplet_model_factory import EfficientNetB0TripletModelFactory
import tensorflow as tf


class MobileNetTripletModelFactory(EfficientNetB0TripletModelFactory):
    def set_basemodel(self):
        mobile_net = tf.keras.applications.MobileNet(input_shape=self.input_shape,
                                                     include_top=False,
                                                     weights="imagenet")

        mobile_net.trainable = False
        return mobile_net


class MobileNetV2TripletModelFactory(MobileNetTripletModelFactory):
    def set_basemodel(self):
        mobile_net = tf.keras.applications.MobileNetV2(input_shape=self.input_shape,
                                                       include_top=False,
                                                       weights="imagenet")

        mobile_net.trainable = False
        return mobile_net
