import tensorflow as tf


class OptimisticTripletMetric:

    def __init__(self):
        pass

    def score(self, y_true, y_pred):
        pos_distance, neg_distance = self.get_distances(y_true, y_pred)

        if tf.reduce_mean(pos_distance) < tf.reduce_mean(neg_distance):
            return 1
        else:
            return 0

    @staticmethod
    def get_distances(y_true, y_pred):
        tmp_pred = tf.broadcast_to(y_pred, (y_pred.shape[0], y_pred.shape[0], y_pred.shape[1]))
        tmp_true = tf.broadcast_to(y_true, (y_true.shape[0], y_true.shape[0]))

        difference = tmp_pred - tf.transpose(tmp_pred, [1, 0, 2])
        difference = tf.norm(difference, axis=2)

        positive_map = tf.cast(tmp_true == tf.transpose(tmp_true, [1, 0]), tmp_pred.dtype)
        # Remove the diagonal entries from the positive map
        positive_map = tf.linalg.set_diag(positive_map,
                                          tf.cast(tf.broadcast_to([0], [positive_map.shape[0]]), tmp_pred.dtype))
        # Remove the symmetric part of the tensor:
        positive_map = tf.linalg.band_part(positive_map, 0, -1)

        # Do the same for the negative examples:
        negative_map = tf.cast(tmp_true != tf.transpose(tmp_true, [1, 0]), tmp_pred.dtype)
        negative_map = tf.linalg.band_part(negative_map, 0, -1)

        pos_distance = tf.boolean_mask(difference, tf.cast(positive_map, tf.bool))
        neg_distance = tf.boolean_mask(difference, tf.cast(negative_map, tf.bool))
        return pos_distance, neg_distance


class PessimisticTripletMetric(OptimisticTripletMetric):

    def score(self, y_true, y_pred):
        pos_distance, neg_distance = self.get_distances(y_true, y_pred)

        if tf.reduce_max(pos_distance) < tf.reduce_min(neg_distance):
            return 1
        else:
            return 0


class AllTripletMetric(OptimisticTripletMetric):

    def score(self, y_true, y_pred):
        pos_distance, neg_distance = self.get_distances(y_true, y_pred)

        if (pos_distance.shape[0] != None) & (neg_distance.shape[0] != None):
            tmp_pos = tf.broadcast_to(pos_distance, (neg_distance.shape[0], pos_distance.shape[0]))
            tmp_neg = tf.transpose(tf.broadcast_to(neg_distance, (pos_distance.shape[0], neg_distance.shape[0])))
            pos_is_smaller = tmp_pos < tmp_neg
            return tf.reduce_sum(tf.cast(pos_is_smaller, tf.int64)) / (pos_is_smaller.shape[0] * pos_is_smaller.shape[1])
        else:
            return 0.
