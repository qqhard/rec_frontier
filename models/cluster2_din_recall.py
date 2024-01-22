import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU
import numpy as np

print(tf.executing_eagerly())


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.item_embedding_table = Embedding(input_dim=1000000,
                                              output_dim=128,
                                              embeddings_initializer='uniform')
        self.cluster_size = 100
        self.cluster_embedding_table = Embedding(input_dim=self.cluster_size,
                                                 output_dim=128,
                                                 embeddings_initializer='uniform')

    # @tf.function
    def __call__(self, inputs):
        self.cluster_embedding = self.cluster_embedding_table(tf.Variable([i for i in range(self.cluster_size)]))

        user_seq = tf.slice(inputs['watch_seq_ids'], [0, 0], [-1, 100])
        user_seq = self.item_embedding_table(user_seq)

        seq_mask = tf.sequence_mask(inputs['watch_seq_len'], maxlen=100, dtype=tf.float32)
        item_info = self.item_embedding_table(inputs['mid'])

        # print(inputs['mid'][0],inputs['watch_seq_ids'][0][0:5] )

        # 聚类
        item_cluster_mat = tf.matmul(tf.stop_gradient(item_info), self.cluster_embedding,
                                     transpose_b=True)  # [B, cluster_size]
        item_top_cluster = tf.nn.top_k(item_cluster_mat, k=1)
        item_top_cluster_emb = self.cluster_embedding_table(tf.reshape(item_top_cluster.indices, [-1, 1]))

        cluster_loss_fn = tf.keras.losses.MeanSquaredError()
        cluster_loss = cluster_loss_fn(tf.stop_gradient(item_info), item_top_cluster_emb)
        # 召回
        item_top_cluster_emb = tf.stop_gradient(item_top_cluster_emb)

        item_top_cluster_emb = tf.reshape(item_top_cluster_emb, [-1, 128])
        cluster_attetion = tf.matmul(user_seq, tf.expand_dims(item_top_cluster_emb, axis=-1))
        cluster_attetion = tf.squeeze(cluster_attetion, axis=-1) - (1 - seq_mask) * 10000
        cluster_attetion = tf.nn.softmax(cluster_attetion, axis=-1)
        user_seq_cluster_din = tf.matmul(user_seq, tf.expand_dims(cluster_attetion, axis=-1), transpose_a=True)
        user_seq_cluster_din = tf.squeeze(user_seq_cluster_din, axis=-1)

        dot = tf.matmul(user_seq_cluster_din, item_info, transpose_b=True)
        pred = tf.nn.softmax(dot, axis=-1)
        labels = tf.eye(item_info.get_shape()[0])

        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(labels, pred)

        return [loss, cluster_loss]

    def get_user_embedding(self, inputs):
        cluster_embedding = self.cluster_embedding_table(tf.Variable([i for i in range(self.cluster_size)]))

        user_seq = tf.slice(inputs['watch_seq_ids'], [0, 0], [-1, 100])
        user_seq = self.item_embedding_table(user_seq)
        seq_mask = tf.sequence_mask(inputs['watch_seq_len'], maxlen=100, dtype=tf.float32)

        user_cluster_mat = tf.matmul(user_seq, cluster_embedding, transpose_b=True)
        user_cluster_mat = tf.nn.softmax(user_cluster_mat, axis=-1)
        user_cluster_score = tf.reduce_sum(user_cluster_mat * tf.expand_dims(seq_mask, axis=-1), axis=1)

        user_top_cluster = tf.math.top_k(user_cluster_score, k=1)

        user_top_cluster_emb = self.cluster_embedding_table(tf.reshape(user_top_cluster.indices, [-1, 1]))
        user_top_cluster_emb = tf.reshape(user_top_cluster_emb, [-1, 128])
        cluster_attetion = tf.matmul(user_seq, tf.expand_dims(user_top_cluster_emb, axis=-1))
        cluster_attetion = tf.squeeze(cluster_attetion, axis=-1) - (1 - seq_mask) * 10000
        cluster_attetion = tf.nn.softmax(cluster_attetion, axis=-1)
        user_seq_cluster_din = tf.matmul(user_seq, tf.expand_dims(cluster_attetion, axis=-1), transpose_a=True)
        user_seq_cluster_din = tf.squeeze(user_seq_cluster_din, axis=-1)

        return user_seq_cluster_din

    def get_item_embedding(self, inputs):
        item_info = self.item_embedding_table(inputs['mid'])
        return item_info
