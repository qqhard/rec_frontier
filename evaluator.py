import tensorflow as tf
def eval_two_tower(model, test_data, metric_names, k=10, batch_size=None, trace=''):
    user_embedding = model.get_user_embedding(test_data)
    item_embedding = model.get_item_embedding(test_data)
    pred = tf.matmul(user_embedding, item_embedding, transpose_b = True)
    user_true_pred = tf.reduce_sum(tf.multiply(pred, tf.eye(pred.get_shape()[0])), axis=-1)
    user_item_pred = pred - tf.expand_dims(user_true_pred, axis=1)

    pos_rank = tf.reduce_sum(tf.cast(tf.greater(user_item_pred, 0), tf.int32), axis=1)
    hitratio = tf.reduce_mean(tf.cast(tf.less_equal(pos_rank, 10), tf.float32))

    return hitratio