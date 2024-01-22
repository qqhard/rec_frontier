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
                                     transpose_b=True) / 5 # [B, cluster_size]
        sample_size = 4
        item_cluster_pro = tf.nn.softmax(item_cluster_mat, axis=-1)  # [B, cluster_size]
        item_cluster_sample = tf.compat.v1.multinomial(tf.math.log(item_cluster_mat), sample_size)  # [B, S]
        item_cluster_sample_label = tf.one_hot(
            tf.reshape(item_cluster_sample, [-1]), self.cluster_size)

        item_sample_cluster_emb = self.cluster_embedding_table(item_cluster_sample)
        user_seq_sample_score = tf.matmul(user_seq, item_sample_cluster_emb,
                                          transpose_b=True)  # [B, seq_size, sample_size]
        user_seq_sample_score = tf.transpose(user_seq_sample_score, perm=[0, 2, 1])  # [B, sample_size, seq_size]
        user_seq_sample_score = user_seq_sample_score - (1 - tf.expand_dims(seq_mask, axis=1)) * 10000
        user_seq_sample_score = tf.nn.softmax(user_seq_sample_score, axis=-1)  # [B, sample_size, seq_size]
        user_sample_embedding = tf.matmul(user_seq_sample_score, user_seq)  # [B, sample_size, embedding_size]
        user_sample_item_score = tf.squeeze(tf.matmul(user_sample_embedding, tf.expand_dims(item_info, axis=-1)),
                                            axis=-1)  # [B, sample_size]
        sample_reward_baseline = tf.reduce_mean(user_sample_item_score, axis=-1, keepdims=True)
        sample_reward = user_sample_item_score - sample_reward_baseline
        sample_reward = tf.reshape(sample_reward, [-1])

        cluster_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        item_sample_cluster_pro = tf.reshape(tf.tile(tf.expand_dims(item_cluster_pro, axis=1), [1, sample_size, 1]),
                                             [-1, self.cluster_size])
        cluster_loss = cluster_loss_fn(item_cluster_sample_label, item_sample_cluster_pro, sample_weight=sample_reward)
        # print(cluster_loss)

        # 召回
        item_top_cluster = tf.nn.top_k(item_cluster_mat, k=1)
        top_indice = item_top_cluster.indices

        item_top_cluster_emb = tf.stop_gradient(
            self.cluster_embedding_table(tf.reshape(item_top_cluster.indices, [-1, 1])))

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
