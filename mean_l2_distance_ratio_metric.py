import tensorflow as tf

def mean_l2_distance_ratio_metric(y_true, y_pred):
    tmp_pred = tf.broadcast_to( y_pred, (y_pred.shape[0], y_pred.shape[0], y_pred.shape[1]) )
    tmp_true = tf.broadcast_to( y_true, (y_true.shape[0], y_true.shape[0]) )

    difference = tmp_pred - tf.transpose(tmp_pred, [1,0,2])
    difference = tf.norm(difference, axis=2)

    positive_map = tf.cast( tmp_true == tf.transpose( tmp_true, [1,0] ), tmp_pred.dtype )
    negative_map = tf.cast( tmp_true != tf.transpose( tmp_true, [1,0]), tmp_pred.dtype )

    sum_positive = tf.reduce_sum( tf.math.multiply( positive_map, difference ) )
    sum_negative = tf.reduce_sum( tf.math.multiply( negative_map, difference ) )

    count_positive = tf.reduce_sum(positive_map) - positive_map.shape[0]
    count_negative = tf.reduce_sum(negative_map)

    mean_pos_dist = sum_positive/count_positive
    mean_neg_dist = sum_negative/count_negative
    if mean_pos_dist < mean_neg_dist:
        return 1
    else:
        return 0