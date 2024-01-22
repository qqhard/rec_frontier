import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU
import numpy as np
print(tf.executing_eagerly())

train_sample_path = '/Users/bytedance/dataset/ml-latest/train_set.csv'
test_sample_path = '/Users/bytedance/dataset/ml-latest/test_set.csv'


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
TEST_BATCH_SIZE = 1000

raw_dataset = tf.data.TFRecordDataset(train_sample_path)
raw_dataset = raw_dataset.shuffle(100000)
parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset = parsed_dataset.batch(BATCH_SIZE, drop_remainder=True).repeat(1)

raw_test_dataset = tf.data.TFRecordDataset(test_sample_path)
raw_test_dataset = raw_test_dataset.shuffle(1000000)
test_parsed_dataset = raw_test_dataset.map(_parse_function).repeat(100)
test_parsed_dataset = test_parsed_dataset.batch(TEST_BATCH_SIZE)






from models.multi_pooling_recall import PoolingRecallModule
model = PoolingRecallModule()
import evaluator

def test_one():
    step = 0

    test_iter= iter(test_parsed_dataset)
    test_inputs = next(test_iter)

    for inputs in parsed_dataset:
        with tf.GradientTape() as tape:
            loss_list = model(inputs)

            # print(loss.numpy())
        optimizer = tf.keras.optimizers.Adagrad(0.1)
        optimizer.minimize(loss_list, var_list=model.trainable_variables, tape=tape)
        if step % 10 == 0:
            hr = evaluator.eval_two_tower(model, test_inputs, metric_names=['hr'], trace='test')
            print("step {} hr : {}".format(step, hr))
            pass

        step += 1
    hr = evaluator.eval_two_tower(model, test_inputs, metric_names=['hr'], trace='test')
    print("final step {} hr : {}".format(step, hr))

test_one()

# test_one()