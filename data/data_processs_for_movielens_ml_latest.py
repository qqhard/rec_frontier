import csv
import tensorflow as tf

train_sample_path = '/Users/bytedance/dataset/ml-latest/train_set.csv'
test_sample_path = '/Users/bytedance/dataset/ml-latest/test_set.csv'

feature_description = {
    'uid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'mid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'watch_seq_ids': tf.io.FixedLenSequenceFeature([100], tf.int64, default_value=0, allow_missing=True),
    'watch_seq_len': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def data_prepare():
    movies_map = {}

    with open('/Users/bytedance/dataset/ml-latest/movies.csv') as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            mid, title, cates = row
            movies_map[mid] = row

    user_behaviors = {}

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_example(uid, mid, watch_seq_ids):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        seq_len = len(watch_seq_ids)
        if len(watch_seq_ids) < 100:
            watch_seq_ids = watch_seq_ids + [0] * (100 - len(watch_seq_ids))
        feature = {
            'uid': _int64_feature(int(uid)),
            'mid': _int64_feature(int(mid)),
            'watch_seq_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=watch_seq_ids)),
            'watch_seq_len': _int64_feature(seq_len)
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    with open('/Users/bytedance/dataset/ml-latest/ratings.csv') as f:
        r = csv.reader(f)
        header = next(r)
        cnt = 0
        for row in r:
            uid, mid, rating, ts = row
            if uid not in user_behaviors:
                user_behaviors[uid] = []
            user_behaviors[uid].append([mid, float(rating), float(ts)])
            cnt += 1
            # if cnt > 10000:
            #     break

        with tf.io.TFRecordWriter(train_sample_path) as train_writer, tf.io.TFRecordWriter(
                test_sample_path) as test_writer:

            for uid, behaviors in user_behaviors.items():
                behaviors = sorted(behaviors, key=lambda x: x[2], reverse=True)
                samples = []
                split_time = behaviors[max(0, int(len(behaviors) * 0.3))][2]
                for index in range(len(behaviors)):
                    target = behaviors[index]
                    seq = list(map(lambda x: int(x[0]), behaviors[index + 1:index + 101]))
                    samples.append([uid, target[0], seq, target[2]])

                for sample in samples:
                    print(sample)
                    example = serialize_example(sample[0], sample[1], sample[2])
                    if sample[3] <= split_time:
                        train_writer.write(example)
                    else:
                        test_writer.write(example)


def data_test():
    raw_dataset = tf.data.TFRecordDataset(train_sample_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    for raw_record in raw_dataset.take(10):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)


data_prepare()
# data_test()
