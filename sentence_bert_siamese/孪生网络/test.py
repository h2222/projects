#coding=utf-8

import tensorflow as tf



# 参数重复使用
with tf.variable_scope(None, default_name='test_scope'):
    v1 = tf.get_variable('a', [1])

with tf.variable_scope('test_scope', reuse=True):
    v2 = tf.get_variable('a', [1])

print(v1)
print(v2)
print(v1 is v2) # 名称不同, 但指向同一个内存区域
print(id(v1) == id(v2))


# 参数服用的嵌套使用, 可以继承scope中的reuse


with tf.variable_scope(default_name='test_scope_1'):
    v1 = tf.get_variable('v1', [1])
    with tf.variable_scope(default_name='sub_test_scope_1'):
        sub_v1 = tf.get_variable('sub_v1', [2])


with tf.variable_scope(default_name='test_scope_1', reuse=True):
    v2 = tf.get_variable('v1', [1])
    with tf.variable_scope(default_name='sub_test_scope_1'):
        sub_v2 = tf.get_variable('sub_v1', [2])


print(v1 is v2)
print(sub_v1 is sub_v2)