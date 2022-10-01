# interact -p GPU-shared --gres=gpu:v100-16:1 -N 1
# cd /ocean/projects/bio210019p/stevesho/data/preprocess/tfrecords/train

import tensorflow as tf

# set mp_type
mp_type = tf.float16

# random labels with -1 to mask
labels = [[2.230741  , 0.68080056, 0.6166089 , -1], [2.230741  , 0.68080056, 0.6166089 , -1]]

# random preds
logits = [[1.22321, 2.33434, 1.23342, 4.43221], [1.22321, 2.33434, 1.23342, 4.43221]]

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


