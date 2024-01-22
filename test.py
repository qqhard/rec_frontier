import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU

print(tf.executing_eagerly())

train_sample_path = '/Users/bytedance/dataset/ml-latest/train_set.csv'

raw_dataset = tf.data.TFRecordDataset(train_sample_path)

feature_description = {
    'uid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'mid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'watch_seq_ids': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
    'watch_seq_len': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


BATCH_SIZE = 128

raw_dataset = raw_dataset.shuffle(10000)
parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset = parsed_dataset.batch(BATCH_SIZE).repeat(100)


class MLP(Layer):
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., use_batch_norm=False):
        """Multilayer Perceptron.
        Args:
            :param hidden_units: A list. The list of hidden layer units's numbers.
            :param activation: A string. The name of activation function, like 'relu', 'sigmoid' and so on.
            :param dnn_dropout: A scalar. The rate of dropout .
            :param use_batch_norm: A boolean. Whether using batch normalization or not.
        :return:
        """
        super(MLP, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.use_batch_norm = use_batch_norm
        self.bt = BatchNormalization()

    def call(self, inputs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        if self.use_batch_norm:
            x = self.bt(x)
        x = self.dropout(x)
        return x


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.user_embedding_table = Embedding(input_dim=1000000,
                                              output_dim=128,
                                              embeddings_initializer='uniform')

        self.item_embedding_table = Embedding(input_dim=1000000,
                                              output_dim=128,
                                              embeddings_initializer='uniform')
        self.item_tower = MLP([128, 96, 64])
        self.user_tower = MLP([128, 96, 64])

    # @tf.function
    def __call__(self, inputs):
        user_info = self.user_embedding_table(inputs['uid'])
        user_seq = tf.slice(inputs['watch_seq_ids'], [0, 0], [-1, 5])
        user_seq = self.item_embedding_table(user_seq)

        seq_mask = tf.sequence_mask(inputs['watch_seq_len'], maxlen=5, dtype=tf.float32)
        user_seq_pooling = tf.reduce_mean(user_seq * tf.expand_dims(seq_mask, axis=-1), axis=1)
        item_info = self.item_embedding_table(inputs['mid'])

        target_attetion = tf.matmul(user_seq, tf.expand_dims(item_info, axis=-1))
        target_attetion = tf.squeeze(target_attetion, axis=-1) - (1 - seq_mask) * 10000
        target_attetion = tf.nn.softmax(target_attetion, axis=-1)
        user_seq_din = tf.matmul(user_seq, tf.expand_dims(target_attetion, axis=-1), transpose_a=True)
        user_seq_din = tf.squeeze(user_seq_din, axis=-1)
        dot = tf.matmul(user_info + user_seq_din, item_info, transpose_b=True)
        pred = tf.nn.softmax(dot, axis=-1)
        print(inputs['mid'][0],inputs['watch_seq_ids'][0][0:5] )

        return pred


model = SimpleModule()

for inputs in parsed_dataset:
    with tf.GradientTape() as tape:
        pred = model(inputs)
        labels = tf.eye(BATCH_SIZE)

        cce = tf.keras.losses.CategoricalCrossentropy()
        loss = cce(labels, pred)
        print(loss.numpy())
    optimizer = tf.keras.optimizers.SGD(10)
    optimizer.minimize(loss, var_list=model.trainable_variables, tape=tape)
