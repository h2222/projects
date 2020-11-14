# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim
uid2goods = "u2g"
goods2goods = "g2g"

mall_rec_u2g_key = "algo_mall_u2g:%d"
mall_rec_g2g_key = "algo_mall_g2g:%d"

b2c_algo_recall_key = "algo_b2c_%s:%s:%d"

rds_host = "r-d7oven45vpwhoe42ev.redis.eu-west-1.rds.aliyuncs.com"
rds_port = 6379
# test env
# rds_host = "10.52.176.96"
# rds_port = 6379
expire_time = 86400 * 7

default_user_id = 1

feature_description = {
    "y": tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    "kyc_level": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "user_create_time": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "gender": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "state": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "city": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "lga": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "category_id": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "brand_id": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "goods_number": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "sell_volume": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "is_new": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "retail_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "extra_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "unit_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    #
    "n0": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n1": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n2": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n3": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n4": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n5": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n6": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n7": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n8": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "n9": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "k0": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "k1": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "k2": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "k3": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "k4": tf.FixedLenFeature(shape=[1], dtype=tf.string),
}

small_embedding_size = 4
bucket_size = 100
max_bucket_size = 1000
initializer = tf.truncated_normal_initializer

good_name_dict = [
    tf.feature_column.categorical_column_with_hash_bucket('n0', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n1', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n2', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n3', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n4', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n5', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n6', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n7', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n8', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n9', max_bucket_size),
]

good_keywords_dict = [
    tf.feature_column.categorical_column_with_hash_bucket('k0', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k1', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k2', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k3', max_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k4', max_bucket_size),
]


# goods_shared_name_dict = tf.feature_column\
#     .shared_embedding_columns(good_name_dict, 10,
#                               shared_embedding_collection_name="goods_shared_name")
#
# goods_shared_keyword_dict = tf.feature_column\
#     .shared_embedding_columns(good_keywords_dict, 10,
#                               shared_embedding_collection_name="goods_shared_keyword")

# shared_feature = goods_shared_name_dict + goods_shared_keyword_dict

indicator_dict = [
    tf.feature_column.categorical_column_with_hash_bucket('gender', 5),
    tf.feature_column.categorical_column_with_hash_bucket('state', bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('city', bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('lga', bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('kyc_level', bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('category_id', bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('brand_id', bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('is_new', bucket_size, dtype=tf.int64),
]

tf_data_columns = [
    tf.feature_column.numeric_column('user_create_time', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('retail_price', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('extra_price', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('unit_price', normalizer_fn=slim.batch_norm, default_value=0.0),
]

int_tf_data_columns = [
    tf.feature_column.numeric_column('goods_number'),
    tf.feature_column.numeric_column('sell_volume'),
]

tf_embedding_columns = [
    tf.feature_column.embedding_column(x, dimension=small_embedding_size,
                                       initializer=initializer)
    for x in indicator_dict + good_name_dict + good_keywords_dict
]

tf_indicator_columns = [
    tf.feature_column.bucketized_column(x, boundaries=[1, 2, 3, 5, 10, 20, 50, 100, 1000, 2000, 5000, 10000]) for x in tf_data_columns
]

tf_int_embedding = [
    tf.feature_column.embedding_column(tf.feature_column.bucketized_column(x, boundaries=[1, 2, 3, 5, 10, 20, 50, 100, 1000, 2000, 5000, 10000]),
                                       dimension=small_embedding_size, initializer=initializer) for x in
    int_tf_data_columns
]

min_user_active = 1
