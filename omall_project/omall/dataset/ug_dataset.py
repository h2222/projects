# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import sys
import time
from consts import *
from datetime import datetime, timedelta
import os


def query_valid_goods_ids():
    # `is_secKill` int(11) DEFAULT '1' COMMENT '商品类型,1:普通商品,2:秒杀,3:团购,4:砍价',
    # 防止推荐结果集中对于秒杀的热门商品进行过滤。对于同一款商品，普通与秒杀在数据库分别对应两条数据。
    goods_query = """
    select id, category_id from
    otrade_dw.dwd_otrade_b2c_mall_nideshop_goods_hf where dt >= '{dt}' and is_delete=0 and is_on_sale=1
    and is_seckill=1
    """
    yt = time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    data = dict(spark.sql(goods_query.format(dt=yt)).rdd.map(lambda x: (int(x.id), int(x.category_id)))
                .filter(lambda x: x is not None).distinct()
                .collect())
    f_path = "/data/omall_b2c_nlp/b2c_valid_goods_%s.txt" % yt
    if os.path.exists(f_path):
        os.system("rm -rf %s" % f_path)
    f = open(f_path, "aw")
    f.write(json.dumps(data))
    f.close()
    # os.system("/usr/lib/hadoop-current/bin/hdfs dfs -put %s /tmp/research/b2c_valid_goods_%s.txt" % (f_path, yt))


def reducer(a, b):
    res = [0] * len(a)
    for x in range(len(a)):
        res[x] = a[x] + b[x]
    return res


def opay_data_dataset(start_dt, days=30):
    data = sc.parallelize([])
    dt = datetime.strptime(start_dt, '%Y-%m-%d')
    for i in range(days):
        try:
            cur_dt = dt - timedelta(days=i)
            print cur_dt
            ret = sc.textFile("oss://opay-datalake/algo_migration/omall/opg_pair/dt=%s/*" % cur_dt.strftime('%Y-%m-%d'))
            data = data.union(ret)
        except:
            pass
    print data.count()
    data.map(json.loads)\
        .map(lambda x: ((int(x["user_id"]), int(x["good_id"])), [int(x["click_num"]), int(x["show_num"])]))\
        .filter(lambda x: x is not None and x[0][0] != 1).reduceByKey(reducer)\
        .map(lambda ((uid, gid), (cnum, snum)): json.dumps({
            "user_id": uid,
            "good_id": gid,
            "click_num": cnum,
            "show_num": snum,
        })).repartition(1).saveAsTextFile("/tmp/omall/opg_pair_dataset/dt=%s" % start_dt)


if __name__ == "__main__":
    spark, sc = init_spark("omall_dataset")
    sc.setLogLevel("ERROR")
    sc_candidate_event_name = sc.broadcast(candidate_event_name)
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    day_range = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print day, day_range
    opay_data_dataset(day, day_range)
    query_valid_goods_ids()
