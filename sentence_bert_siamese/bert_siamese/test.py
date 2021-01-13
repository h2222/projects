import tensorflow as tf



ts_1 = tf.placeholder(shape=[7, 10, 128], dtype=tf.float32)
ts_2 = tf.placeholder(shape=[7, 10, 128], dtype=tf.float32)
ts_mask = tf.ones(shape=[10], dtype=tf.float32)

mul_mask = lambda x, m: (x * tf.expand_dims(m, axis=-1))[:, 1:, :] 

masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10 - 1)

tensor_a = tf.reduce_max(ts_1, axis=1)
tensor_b = tf.reduce_max(ts_2, axis=1)


print(tensor_a.shape)
print(tensor_b.shape)
print(tf.abs(tensor_a - tensor_b).shape)
print((tensor_a * tensor_b).shape)


# 4 * [7, 128] -> [7, 128 * 4]
output_layer = tf.concat([tensor_a, tensor_b, tf.abs(tensor_a - tensor_b), tensor_a * tensor_b], axis=1)


#[7, 128 * 4] -> [7, 1]
logits = tf.layers.dense(
                    output_layer,
                    1,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                    )

# print(output_layer)
# print(logits)






import tensorflow as tf
import numpy as np

X = np.random.randn(2, 10, 8) # [2, 10, 8]
# The second example is of length 6
X[1, 6:] = 0
X_lengths = [10, 6] # [10, 6]

# X[b, 10, 8] * Wx[8, 5] + H[b, 10, 5] + Wh[5, 5] + b[5, 1] ---> output[b, 10, 5]\
# lstm cell 指 lstm 的层数=10, lstm unit 指 每个cell中全连接层的个数=5
cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True) 

outputs, states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell, cell_bw=cell, dtype=tf.float64, sequence_length=X_lengths, inputs=X
)

output_fw, output_bw = outputs
states_fw, states_bw = states

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    states_shape = tf.shape(states)
    print(states_shape.eval())
    c, h = states_fw
    o = output_fw


    print("c shape:", c)
    print("h shape:", h.shape)
    print("o shape:", o)


