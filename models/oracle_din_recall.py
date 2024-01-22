import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU
import numpy as np

print(tf.executing_eagerly())


class OracleDinRecallModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        # self.user_embedding_table = Embedding(input_dim=1000000,
        #                                       output_dim=128,
        #                                       embeddings_initializer='uniform')

        self.item_embedding_table = Embedding(input_dim=1000000,
                                              output_dim=128,
                                              embeddings_initializer='uniform')

    # @tf.function
    def __call__(self, inputs):
        # user_info = self.user_embedding_table(inputs['uid'])
        user_seq = tf.slice(inputs['watch_seq_ids'], [0, 0], [-1, 100])
        user_seq = self.item_embedding_table(user_seq)

        seq_mask = tf.sequence_mask(inputs['watch_seq_len'], maxlen=100, dtype=tf.float32)
        user_seq_pooling = tf.reduce_mean(user_seq * tf.expand_dims(seq_mask, axis=-1), axis=1)
        item_info = self.item_embedding_table(inputs['mid'])

        target_attetion = tf.matmul(user_seq, tf.expand_dims(item_info, axis=-1))
        target_attetion = tf.squeeze(target_attetion, axis=-1) - (1 - seq_mask) * 10000
        target_attetion = tf.nn.softmax(target_attetion, axis=-1)
        user_seq_din = tf.matmul(user_seq, tf.expand_dims(target_attetion, axis=-1), transpose_a=True)
        user_seq_din = tf.squeeze(user_seq_din, axis=-1)

        # print(inputs['mid'][0],inputs['watch_seq_ids'][0][0:5] )

        dot = tf.matmul(user_seq_din, item_info, transpose_b=True)
        pred = tf.nn.softmax(dot, axis=-1)

        labels = tf.eye(item_info.get_shape()[0])

        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(labels, pred)

        return [loss]

    def get_user_embedding(self, inputs):
        # user_info = self.user_embedding_table(inputs['uid'])

        user_seq = tf.slice(inputs['watch_seq_ids'], [0, 0], [-1, 100])
        user_seq = self.item_embedding_table(user_seq)
        seq_mask = tf.sequence_mask(inputs['watch_seq_len'], maxlen=100, dtype=tf.float32)

        item_info = self.item_embedding_table(inputs['mid'])
        target_attetion = tf.matmul(user_seq, tf.expand_dims(item_info, axis=-1))
        target_attetion = tf.squeeze(target_attetion, axis=-1) - (1 - seq_mask) * 10000
        target_attetion = tf.nn.softmax(target_attetion, axis=-1)
        user_seq_din = tf.matmul(user_seq, tf.expand_dims(target_attetion, axis=-1), transpose_a=True)
        user_seq_din = tf.squeeze(user_seq_din, axis=-1)

        return user_seq_din

    def get_item_embedding(self, inputs):
        item_info = self.item_embedding_table(inputs['mid'])
        return item_info
