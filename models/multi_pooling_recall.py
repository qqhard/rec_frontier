import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU
import numpy as np

print(tf.executing_eagerly())


class MLP(Layer):
    def __init__(self, hidden_units, activation='relu'):
        """Multilayer Perceptron.
        Args:
            :param hidden_units: A list. The list of hidden layer units's numbers.
            :param activation: A string. The name of activation function, like 'relu', 'sigmoid' and so on.
            :param dnn_dropout: A scalar. The rate of dropout .
            :param use_batch_norm: A boolean. Whether using batch normalization or not.
        :return:
        """
        super(MLP, self).__init__()
        self.dnn_network = []
        for i in range(len(hidden_units)):
            if i < len(hidden_units) - 1:
                self.dnn_network.append(Dense(units=hidden_units[i], activation=activation))
            else:
                self.dnn_network.append(Dense(units=hidden_units[i], activation=None))

    def call(self, inputs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        return x


class PoolingRecallModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.item_embedding_table = Embedding(input_dim=1000000,
                                              output_dim=128,
                                              embeddings_initializer='uniform')

        self.multi_mask = tf.Variable([2 ** i for i in range(7)] + [100])
        self.multi_mask = tf.sequence_mask(self.multi_mask, maxlen=100, dtype=tf.float32)
        self.user_tower_fn = MLP([512, 256, 128])
        self.item_tower_fn = MLP([512, 256, 128])

    # @tf.function
    def __call__(self, inputs):
        user_seq = tf.slice(inputs['watch_seq_ids'], [0, 0], [-1, 100])
        user_seq = self.item_embedding_table(user_seq)
        seq_mask = tf.sequence_mask(inputs['watch_seq_len'], maxlen=100, dtype=tf.float32)
        user_seq = user_seq * tf.expand_dims(seq_mask, axis=-1)

        # [B, emb_size, mask_size]
        multi_user_seq = tf.squeeze(tf.matmul(user_seq, self.multi_mask, transpose_a=True, transpose_b=True))
        # [B, mask_size, emb_size]
        multi_user_seq = tf.transpose(multi_user_seq, perm=[0, 2, 1])
        # [B, mask_size*emb_size]
        multi_user_seq = tf.reshape(multi_user_seq, [-1, multi_user_seq.get_shape()[1] * multi_user_seq.get_shape()[2]])

        user_tower = self.user_tower_fn(multi_user_seq)

        item_info = self.item_embedding_table(inputs['mid'])
        item_tower = self.item_tower_fn(item_info)

        dot = tf.matmul(user_tower, item_tower, transpose_b=True)
        pred = tf.nn.softmax(dot, axis=-1)
        labels = tf.eye(item_info.get_shape()[0])

        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(labels, pred)

        return [loss]

    def get_user_embedding(self, inputs):
        user_seq = tf.slice(inputs['watch_seq_ids'], [0, 0], [-1, 100])
        user_seq = self.item_embedding_table(user_seq)
        seq_mask = tf.sequence_mask(inputs['watch_seq_len'], maxlen=100, dtype=tf.float32)
        user_seq = user_seq * tf.expand_dims(seq_mask, axis=-1)

        # [B, emb_size, mask_size]
        multi_user_seq = tf.squeeze(tf.matmul(user_seq, self.multi_mask, transpose_a=True, transpose_b=True))
        # [B, mask_size, emb_size]
        multi_user_seq = tf.transpose(multi_user_seq, perm=[0, 2, 1])
        # [B, mask_size*emb_size]
        multi_user_seq = tf.reshape(multi_user_seq, [-1, multi_user_seq.get_shape()[1] * multi_user_seq.get_shape()[2]])

        user_tower = self.user_tower_fn(multi_user_seq)

        return user_tower

    def get_item_embedding(self, inputs):
        item_info = self.item_embedding_table(inputs['mid'])
        item_tower = self.item_tower_fn(item_info)
        return item_tower
