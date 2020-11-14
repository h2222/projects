# -*- coding: utf-8 -*-
import os
from consts import *
import oss2


def DataSetBucket(pattern, train_path, epochs, batch_size, fd=feature_description):
    default_shuffle_size = 500000
    def input_fn():
        data_files = [train_path + x for x in os.listdir(train_path)]
        dataset = tf.data.TFRecordDataset(data_files)
        def parser(serialized):
            data = tf.parse_single_example(
                serialized,
                features=fd
            )
            return data
        if pattern == "train":
            data = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .shuffle(buffer_size=default_shuffle_size).repeat(epochs).batch(batch_size) \
                .prefetch(buffer_size=tf.data.experimental.AUTOTUNE).map(lambda x: (x, x['y']))
        else:
            data = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(epochs) \
                .batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).map(lambda x: (x, x['y']))
        return data
    return input_fn


def get_data(dt1, d1, dt2, d2, type_name):
    root_path = "/data/%s" % type_name
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    tmp_dataset_path = "/tmp/research/mall_%s_%s_%d/" % (type_name, dt1, d1)
    os.system("/usr/bin/sh /data/download.sh %s %s" % (tmp_dataset_path, root_path))
    tmp_dataset_path = "/tmp/research/mall_%s_%s_%d/" % (type_name, dt2, d2)
    os.system("/usr/bin/sh /data/download.sh %s %s" % (tmp_dataset_path, root_path))


def upload_estimator_via_oss(model_type, dt):
    timestamp_len = 10
    folder_path = "/data/estimator_%s/" % model_type
    os.chdir(folder_path)
    model_lists = os.listdir(folder_path)
    for m in model_lists:
        if len(m) != timestamp_len:
            os.system("rm -rf %s" % m)
    model_lists.sort(reverse=True)
    if len(model_lists) == 0:
        return
    new_model = model_lists[0]
    if os.path.exists(dt):
        os.system("rm -rf %s" % dt)
    os.system("cp -r %s %s" % (new_model, dt))
    zip_model_name = "%s.tar.gz" % dt
    os.system("tar -zcvf %s %s" % (zip_model_name, dt))
    auth = oss2.Auth('LTAI4FmshDAuao6Wyn1BAoGB', 'MLhLZpdpL1sNrtUNsfcfcFj4eHq7kp')
    bucket = oss2.Bucket(auth, 'http://oss-eu-west-1-internal.aliyuncs.com', 'oride-algo')
    put_key = "tf_model/{model_type}/{model_name}".format(model_type=model_type, model_name=zip_model_name)
    with open(zip_model_name, 'rb') as fileobj:
        bucket.put_object(put_key, fileobj)
    print("Dt: %s | ModelType: %s | Upload to oss" % (dt, model_type))
    os.system("rm -rf %s" % zip_model_name)
    os.system("rm -rf %s" % dt)


def remove_data(dt1, d1, dt2, d2, type_name):
    path1 = "/data/%s/%s/" % (type_name, "mall_%s_%s_%d" % (type_name, dt1, d1))
    path2 = "/data/%s/%s/" % (type_name, "mall_%s_%s_%d" % (type_name, dt2, d2))
    os.system("rm -rf %s" % path1)
    os.system("rm -rf %s" % path2)
    path1 = "/tmp/research/mall_%s_%s_%d/" % (type_name, dt1, d1)
    path2 = "/tmp/research/mall_%s_%s_%d/" % (type_name, dt2, d2)
    os.system("/usr/lib/hadoop-current/bin/hdfs dfs -rm -r -skipTrash %s" % path1)
    os.system("/usr/lib/hadoop-current/bin/hdfs dfs -rm -r -skipTrash %s" % path2)


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    receiver_tensors = {
        "kyc_level": tf.placeholder(tf.int64, [None, 1], name="kyc_level"),
        "user_create_time": tf.placeholder(tf.float32, [None, 1], name="user_create_time"),
        #
        "gender": tf.placeholder(tf.string, [None, 1], name="gender"),
        "state": tf.placeholder(tf.string, [None, 1], name="state"),
        "city": tf.placeholder(tf.string, [None, 1], name="city"),
        "lga": tf.placeholder(tf.string, [None, 1], name="lga"),
        #
        "category_id": tf.placeholder(tf.int64, [None, 1], name="category_id"),
        "brand_id": tf.placeholder(tf.int64, [None, 1], name="brand_id"),
        "goods_number": tf.placeholder(tf.int64, [None, 1], name="goods_number"),
        "sell_volume": tf.placeholder(tf.int64, [None, 1], name="sell_volume"),
        "is_new": tf.placeholder(tf.int64, [None, 1], name="is_new"),
        "retail_price": tf.placeholder(tf.float32, [None, 1], name="retail_price"),
        "extra_price": tf.placeholder(tf.float32, [None, 1], name="extra_price"),
        "unit_price": tf.placeholder(tf.float32, [None, 1], name="unit_price"),
        #
        "n0": tf.placeholder(tf.string, [None, 1], name="n0"),
        "n1": tf.placeholder(tf.string, [None, 1], name="n1"),
        "n2": tf.placeholder(tf.string, [None, 1], name="n2"),
        "n3": tf.placeholder(tf.string, [None, 1], name="n3"),
        "n4": tf.placeholder(tf.string, [None, 1], name="n4"),
        "n5": tf.placeholder(tf.string, [None, 1], name="n5"),
        "n6": tf.placeholder(tf.string, [None, 1], name="n6"),
        "n7": tf.placeholder(tf.string, [None, 1], name="n7"),
        "n8": tf.placeholder(tf.string, [None, 1], name="n8"),
        "n9": tf.placeholder(tf.string, [None, 1], name="n9"),
        #
        "k0": tf.placeholder(tf.string, [None, 1], name="k0"),
        "k1": tf.placeholder(tf.string, [None, 1], name="k1"),
        "k2": tf.placeholder(tf.string, [None, 1], name="k2"),
        "k3": tf.placeholder(tf.string, [None, 1], name="k3"),
        "k4": tf.placeholder(tf.string, [None, 1], name="k4"),
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)
