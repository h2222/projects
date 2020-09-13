#coding=utf-8
import tensorflow as tf


def test():
    tensor1 = tf.range(0, 10, dtype=tf.int32)
    tensor2 = tf.reshape(tensor1, [-1, 1])

    print(tensor1)
    print(tensor2)


def test2(path):

    with open(path, 'rb',) as f:
        for i in f.readlines():
            print(i.decode('utf-8'))


if __name__ == "__main__":
        # test()
        test2('./data/beauty_hao-2020-7-7.vocab')