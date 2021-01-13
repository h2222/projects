# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import sys
import time
from consts import *
import redis
from operator import add

goods_query = """
select id, update_time, category_id, brand_id from
otrade_dw.dwd_otrade_b2c_mall_nideshop_goods_hf where dt = '{dt}' and is_delete=0 and is_on_sale=1
and is_seckill=1
"""

# only with payment
order_query = """
select goods_id from
(select id, opayid, pay_time, __ts_ms from otrade_dw_ods.ods_binlog_mall_nideshop_order_hi
where dt >= '{dt}' and pay_time is not null and pay_time > 0) as t1
join
(select order_id, goods_id from otrade_dw_ods.ods_binlog_mall_nideshop_order_goods_hi
where dt >='{dt}') as t2
on t1.id=t2.order_id
"""


def day_ago(day_num):
    return time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400 * day_num))


def write_ids2rds(key, ids):
    rds = redis.StrictRedis(host=rds_host, port=rds_port)
    rds.set(key, json.dumps({
        "good_ids": ids
    }), ex=expire_time)
    time.sleep(0.1)


def mapper(x):
    try:
        cid = int(x.category_id) if x.category_id is not None else 0
        bid = int(x.brand_id) if x.brand_id is not None else 0
        return int(x.id), (x.update_time, cid, bid)
    except:
        pass


def simple_mapper(x):
    try:
        return int(x.goods_id), 1
    except:
        pass


def hot_sell_recall(dt):
    # 热门点击召回
    raw_data = spark.sql(goods_query.format(dt=dt)).rdd \
        .map(mapper).filter(lambda x: x is not None) \
        .reduceByKey(max) \
        .cache()
    recall_dt = day_ago(30)
    sell_ids = spark.sql(order_query.format(dt=recall_dt)).rdd \
        .map(simple_mapper) \
        .filter(lambda x: x is not None).reduceByKey(add)
    all_hot_sell_rank = sell_ids.repartition(1).sortBy(lambda x: x[-1], ascending=False).collect()
    hot_ids = [x[0] for x in all_hot_sell_rank if x[1] > 1]
    print "hot_sell_ids: %d" % len(hot_ids)
    write_ids2rds(mall_rec_hot_sell_key, hot_ids[:int(len(hot_ids) * 0.5)])
    ctg_res = sell_ids.join(raw_data.filter(lambda (gid, (_, cid, bid)): cid > 0)) \
        .map(lambda (gid, (sell_num, (_, ctg_id, bd_id))): (ctg_id, (gid, sell_num))) \
        .groupByKey().mapValues(list).collect()
    bd_res = sell_ids.join(raw_data.filter(lambda (gid, (_, cid, bid)): bid > 0)) \
        .map(lambda (gid, (sell_num, (_, ctg_id, bd_id))): (bd_id, (gid, sell_num))) \
        .groupByKey().mapValues(list).collect()
    for elem in ctg_res:
        (ctg_id, raw_data) = elem
        raw_data.sort(key=lambda x: x[-1], reverse=True)
        ids = [x[0] for x in raw_data]
        res_ids = ids[:int(len(ids) * 0.5)]
        if len(res_ids) > 0:
            print mall_rec_hot_sell_category_key % ctg_id, len(res_ids)
            write_ids2rds(mall_rec_hot_sell_category_key % ctg_id, res_ids)
    for elem in bd_res:
        (bd_id, raw_data) = elem
        raw_data.sort(key=lambda x: x[-1], reverse=True)
        ids = [x[0] for x in raw_data]
        res_ids = ids[:int(len(ids) * 0.5)]
        if len(res_ids) > 0:
            print mall_rec_hot_sell_brand_key % bd_id, len(res_ids)
            write_ids2rds(mall_rec_hot_sell_brand_key % bd_id, res_ids)


if __name__ == "__main__":
    spark, sc = init_spark("hot_sell_rank")
    sc.setLogLevel("ERROR")
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    print day
    hot_sell_recall(day)
