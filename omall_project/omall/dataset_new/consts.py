# -*- coding: utf-8 -*-
from pyspark.sql.types import *

basic_columns = ['good_id', 'category_id', 'brand_id', 'goods_number', 'sell_volume',
                 'is_new', 'rp_l', 'mp_l', 'kyc_level']
int_stat_basic_columns = ['u_c', 'u_s', 'g_c', 'g_s', 'rpl_s', 'rpl_c', 'rpl_u_s', 'rpl_u_c', 'mpl_s', 'mpl_c',
                          'mpl_u_s', 'mpl_u_c', 'g_ctg_s', 'g_ctg_c', 'g_ctg_u_s',
                          'g_ctg_u_c', 'g_bd_s', 'g_bd_c', 'g_bd_u_s', 'g_bd_u_c']
float_basic_columns = ['user_create_time', 'retail_price', 'extra_price', 'unit_price', 'counter_price', 'market_price']
str_basic_columns = ['name', 'keywords', 'user_id', 'gender', 'state', 'city', 'lga']

fields = [
    StructField("y", FloatType()),
    # user related
    StructField("good_id", IntegerType()),
    StructField("category_id", IntegerType()),
    StructField("brand_id", IntegerType()),
    StructField("goods_number", IntegerType()),
    StructField("sell_volume", IntegerType()),
    StructField("is_new", IntegerType()),
    StructField("rp_l", IntegerType()),
    StructField("mp_l", IntegerType()),
    StructField("kyc_level", IntegerType()),

    StructField("user_create_time", FloatType()),
    StructField("retail_price", FloatType()),
    StructField("extra_price", FloatType()),
    StructField("unit_price", FloatType()),
    StructField("counter_price", FloatType()),
    StructField("market_price", FloatType()),

    StructField("user_id", StringType()),
    StructField("gender", StringType()),
    StructField("state", StringType()),
    StructField("city", StringType()),
    StructField("lga", StringType()),

    StructField("n0", StringType()),
    StructField("n1", StringType()),
    StructField("n2", StringType()),
    StructField("n3", StringType()),
    StructField("n4", StringType()),
    StructField("n5", StringType()),
    StructField("n6", StringType()),
    StructField("n7", StringType()),
    StructField("n8", StringType()),
    StructField("n9", StringType()),
    StructField("k0", StringType()),
    StructField("k1", StringType()),
    StructField("k2", StringType()),
    StructField("k3", StringType()),
    StructField("k4", StringType()),

    StructField("stat_data", ArrayType(FloatType(), False)),
]
