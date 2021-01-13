# -*- coding: utf-8 -*-
import sys
from click_util import *
import pandas as pd
import time
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params, config):
    acti = None
    net = tf.feature_column.input_layer(features, params['data_columns'])
    net1 = tf.feature_column.input_layer(features, params['onehot_columns'])
    net2 = tf.feature_column.input_layer(features, params['embedding_columns'])
    for unit in [2048, 1024, 256, 32]:
        net = tf.layers.dense(net, unit, activation=acti)
        net1 = tf.layers.dense(net1, unit, activation=acti)
        net2 = tf.layers.dense(net2, unit, activation=acti)
    net = tf.concat([net, net1, net2], axis=-1)
    for unit in [512, 256, 64, 8]:
        net = tf.layers.dense(net, unit, activation=acti)
    logits = tf.layers.dense(net, 1, activation=tf.nn.sigmoid)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'output': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = -tf.reduce_mean(
        100 * labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)) + (1 - labels) * tf.log(
            tf.clip_by_value(1 - logits, 1e-10, 1.0)))
    # 计算精确度.
    metrics = {'auc': tf.metrics.auc(labels, logits)}
    # 判断是否为评估
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    # 如果是训练模式
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_model():
    upload_model_type = "mall_" + model_type
    model = tf.estimator.Estimator(
        model_fn=model_fn,  # 制定本模型的模型参数
        params={  # 模型的额外参数，会传递给模型参数的params
            "data_columns": tf_data_columns + int_tf_data_columns,
            "onehot_columns": tf_indicator_columns,
            "embedding_columns": tf_embedding_columns + tf_int_embedding,
        })
    train_path = "/data/%s/%s/" % (model_type, "%s_%s_%d" % (upload_model_type, train_dt, day1))
    test_path = "/data/%s/%s/" % (model_type, "%s_%s_%d" % (upload_model_type, test_dt, day2))
    data = DataSetBucket("train", train_path, 20, 4096 * 2)
    test_data = DataSetBucket("evaluation", test_path, 1, 2048)
    early_stopping = tf.estimator.experimental.stop_if_no_increase_hook(
        model,
        metric_name='auc',
        min_steps=500,
        max_steps_without_increase=1000,
        run_every_secs=None,
        run_every_steps=100)
    tf.estimator.train_and_evaluate(
        model,
        train_spec=tf.estimator.TrainSpec(data, hooks=[early_stopping]),
        eval_spec=tf.estimator.EvalSpec(test_data))
    #
    print "train======"
    single_data = DataSetBucket("train", train_path, 1, 8192)
    result = model.evaluate(single_data)
    print(pd.Series(result))
    print "test======"
    result = model.evaluate(test_data)
    print(pd.Series(result))
    #
    model.export_saved_model(
        "/data/estimator_%s/" % upload_model_type,
        serving_input_receiver_fn,
    )
    upload_estimator_via_oss(upload_model_type, train_dt)


if __name__ == '__main__':
    model_type = "click_model"
    train_dt = sys.argv[1] if len(sys.argv) > 1 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400 * 2))
    day1 = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    test_dt = sys.argv[3] if len(sys.argv) > 3 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    day2 = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print train_dt, day1, test_dt, day2
    get_data(train_dt, day1, test_dt, day2, model_type)
    train_model()
    remove_data(train_dt, day1, test_dt, day2, model_type)
