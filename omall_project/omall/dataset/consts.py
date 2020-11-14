# -*- coding: utf-8 -*-
from pyspark.sql.types import *

candidate_event_name = [
    "TAB_MALL_detailspage_share_clik",
    "TAB_MALL_detailspage_cart_clik",
    "TAB_MALL_detailspage_addtocart_clik",
    "TAB_MALL_detailspage_select_clik",
]

mall_rec_hot_key = "algo_mall_hot"
hot_size = 500
mall_rec_category_key = "algo_mall_ctg:%d"
mall_rec_brand_key = "algo_mall_bd:%d"

max_category_num = 200
max_brand_num = 100

max_recall_rate = 0.5

rds_host = "r-d7oven45vpwhoe42ev.redis.eu-west-1.rds.aliyuncs.com"
rds_port = 6379
# test env
# rds_host = "10.52.176.96"
# rds_port = 6379
expire_time = 86400 * 7

mall_rec_hot_sell_key = "algo_mall_hotsell"
mall_rec_hot_sell_category_key = "algo_mall_hotsell_ctg:%d"
mall_rec_hot_sell_brand_key = "algo_mall_hotsell_bd:%d"
