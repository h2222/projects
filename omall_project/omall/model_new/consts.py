# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.slim as slim

basic_columns = ['good_id', 'category_id', 'brand_id', 'goods_number', 'sell_volume',
                 'is_new', 'rp_l', 'mp_l', 'kyc_level']
int_stat_basic_columns = ['u_c', 'u_s', 'g_c', 'g_s', 'rpl_s', 'rpl_c', 'rpl_u_s', 'rpl_u_c', 'mpl_s', 'mpl_c',
                          'mpl_u_s', 'mpl_u_c', 'g_ctg_s', 'g_ctg_c', 'g_ctg_u_s',
                          'g_ctg_u_c', 'g_bd_s', 'g_bd_c', 'g_bd_u_s', 'g_bd_u_c']
float_basic_columns = ['user_create_time', 'retail_price', 'extra_price', 'unit_price', 'counter_price', 'market_price']
str_basic_columns = ['name', 'keywords', 'user_id', 'gender', 'state', 'city', 'lga']

feature_description = {
    "y": tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    "good_id": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "category_id": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "brand_id": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "goods_number": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "sell_volume": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "is_new": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "rp_l": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "mp_l": tf.FixedLenFeature(shape=[1], dtype=tf.int64),
    "kyc_level": tf.FixedLenFeature(shape=[1], dtype=tf.int64),

    "user_id": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "gender": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "state": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "city": tf.FixedLenFeature(shape=[1], dtype=tf.string),
    "lga": tf.FixedLenFeature(shape=[1], dtype=tf.string),
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

    "user_create_time": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "retail_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "extra_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "unit_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "counter_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),
    "market_price": tf.FixedLenFeature(shape=[1], dtype=tf.float32),

    "stat_data": tf.FixedLenFeature(shape=[len(int_stat_basic_columns)], dtype=tf.float32),

}
small_embedding_size = 3
bucket_size = 100
middle_bucket_size = 500
max_bucket_size = 5000
user_bucket_size = 50000
initializer = tf.truncated_normal_initializer

good_name_dict = [
    tf.feature_column.categorical_column_with_hash_bucket('n0', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n1', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n2', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n3', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n4', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n5', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n6', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n7', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n8', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('n9', middle_bucket_size),
]

good_keywords_dict = [
    tf.feature_column.categorical_column_with_hash_bucket('k0', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k1', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k2', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k3', middle_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('k4', middle_bucket_size),
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
    tf.feature_column.categorical_column_with_hash_bucket('user_id', user_bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('gender', 5),
    tf.feature_column.categorical_column_with_hash_bucket('state', bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('city', bucket_size),
    tf.feature_column.categorical_column_with_hash_bucket('lga', bucket_size),

    tf.feature_column.categorical_column_with_hash_bucket('good_id', max_bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('category_id', max_bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('brand_id', max_bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('is_new', 5, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('rp_l', bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('mp_l', bucket_size, dtype=tf.int64),
    tf.feature_column.categorical_column_with_hash_bucket('kyc_level', bucket_size, dtype=tf.int64),

]
# https://www.cnblogs.com/xianbin7/p/10661572.html
tf_data_columns = [
    tf.feature_column.numeric_column('user_create_time', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('retail_price', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('extra_price', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('unit_price', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('counter_price', normalizer_fn=slim.batch_norm, default_value=0.0),
    tf.feature_column.numeric_column('market_price', normalizer_fn=slim.batch_norm, default_value=0.0),

    tf.feature_column.numeric_column('stat_data', shape=(len(int_stat_basic_columns),),
                                     normalizer_fn=slim.batch_norm),
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
    tf.feature_column.bucketized_column(x, boundaries=[1, 2, 3, 5, 10, 20, 50, 100, 1000, 2000, 5000, 10000]) for x in
    tf_data_columns
]

tf_int_embedding = [
    tf.feature_column.embedding_column(
        tf.feature_column.bucketized_column(x, boundaries=[1, 2, 3, 5, 10, 20, 50, 100, 1000, 2000, 5000, 10000]),
        dimension=small_embedding_size, initializer=initializer) for x in
    int_tf_data_columns
]

min_user_active = 1
