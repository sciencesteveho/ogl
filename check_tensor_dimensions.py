# interact -p GPU-shared --gres=gpu:v100-16:1 -N 1
# cd /ocean/projects/bio210019p/stevesho/data/preprocess/tfrecords/train

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# set mp_type
mp_type = tf.float16

def map_fn(raw_record):
    """Parses a serialized protobuf example into a dictionary
    of input features and labels for Graph training.
    """
    feature_map = {
        "adj": tf.io.FixedLenFeature(
            [2500, 2500], tf.float32
        ),
        "node_feat": tf.io.FixedLenFeature(
            [2500, 33], tf.int64
        ),
        "node_mask": tf.io.FixedLenFeature(
            [2500, 1], tf.float32
        ),
        "label": tf.io.FixedLenFeature([4], tf.float32),
    }

    example = tf.io.parse_single_example(raw_record, feature_map)
    for name in list(example.keys()):
        feature = example[name]
        if feature.dtype == tf.int64:
            feature = tf.cast(feature, tf.int32)
            example[name] = feature

    feature = {
        "adj": tf.cast(example["adj"], mp_type),
        "node_mask": tf.cast(example["node_mask"], mp_type),
    }

    node_feat = example["node_feat"]
    for i in range(33):
        name = "node_feat" + str(i)
        feature[name] = tf.cast(node_feat[..., i], tf.int32)
    label = tf.cast(example["label"], mp_type)

    return (feature, label)

dataset = tf.data.TFRecordDataset('ggraphmutagenesis_1.tfrecords_train_1')
dataset = dataset.map(map_fn)
batched = dataset.batch(32)
batched_numpy = tfds.as_numpy(batched)

# random labels with -1 to mask
labels = [[2.230741  , 0.68080056, 0.6166089 , -1], [2.230741  , 0.68080056, 0.6166089 , -1]]

# random preds
logits = [[1.22321, 2.33434, 1.23342, 4.43221], [1.22321, 2.33434, 1.23342, 4.43221]]


np.random.rand(32, 4)

# conver to tf and cast to mp.type
labels = tf.convert_to_tensor(labels)
logits = tf.convert_to_tensor(logits)
_labels = tf.cast(labels, mp_type)
_logits = tf.cast(logits, mp_type)

# create mask
mask = tf.not_equal(labels, -1)
mask = tf.cast(mask, mp_type)

# get num valid, batch size = 2
num_valid = tf.reduce_sum(mask)
num_valid = tf.cond(
    tf.equal(num_valid, tf.cast(0, mp_type)),
    lambda: tf.cast(1, mp_type),
    lambda: num_valid,
)

num_valid = tf.math.divide(2, num_valid)
_size, _num_targets = tf.shape(labels)[0], tf.shape(labels)[1]
num_valid = tf.broadcast_to(num_valid, [_size,])

# cross entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=_labels, logits=_logits
)
loss = tf.multiply(loss, mask)
loss = tf.reduce_mean(
    tf.reduce_sum(loss, axis=1) * tf.cast(num_valid, mp_type)
)
loss = tf.cast(loss, logits.dtype)  # <tf.Tensor: shape=(), dtype=float32, numpy=0.10644531>


#RMSE 
loss2 = tf.math.squared_difference(_logits, _labels)
loss2 = tf.math.sqrt(loss2)
loss2 = tf.multiply(loss2, mask)
loss2 = tf.reduce_mean(
    tf.reduce_sum(loss2, axis=1) * tf.cast(num_valid, mp_type)
)
loss2 = tf.cast(loss2, logits.dtype)  # <tf.Tensor: shape=(), dtype=float32, numpy=1.0917969>


