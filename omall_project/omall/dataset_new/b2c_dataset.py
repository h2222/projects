# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import sys
import time
from consts import *
from datetime import datetime, timedelta

left_bucket = "<"
right_bucket = ">"


def filter_special_str(x):
    res = ""
    for s in x:
        if s not in (",", "(", ")", '"', "/", ":", "&", "-",
                     '\\', "{", "}", "+", "【", "】", "*", ";", '.', left_bucket,
                     right_bucket, "=", '[', "]"):
            res += s
        else:
            res += " "
    return res


def special_str(x, target_len=10):
    n_list_tmp = [i.lower() for i in filter_special_str(x).split(" ") if len(i) > 1]
    n_list = []
    for e in n_list_tmp:
        if e not in n_list:
            n_list.append(e)
    n_len = len(n_list)
    if n_len < target_len:
        n_list += [""] * (target_len - n_len)
    return n_list[:target_len]


def mapper1(x):
    try:
        return json.loads(x)
    except:
        pass


def mapper2(rec):
    try:
        good_name = rec['str_data'][0]
        good_keyword = rec['str_data'][1]
        return [float(rec['label'])] + rec['basic_data'] + rec['float_data'] + \
               rec['str_data'][2:] + special_str(good_name) + special_str(good_keyword, 5) + \
               [map(float, rec['stat_data'])]
    except:
        pass


def get_dataset(start_dt, days=7):
    schema = StructType(fields)
    data = sc.parallelize([])
    dt = datetime.strptime(start_dt, '%Y-%m-%d')
    for i in range(days):
        try:
            cur_dt = dt - timedelta(days=i)
            ret = sc.textFile(
                "oss://opay-datalake/algo_migration/omall/%s/dt=%s/*" % (model_type, cur_dt.strftime('%Y-%m-%d')))
            data = data.union(ret)
            print cur_dt
        except:
            pass
    data = data.map(mapper1).filter(lambda x: x is not None)
    # if days > 2:
    #     print data.count()
    #     user_stat = data.map(lambda x: (int(x["user_id"]), int(x["label"])))\
    #         .reduceByKey(add)
    #     print "all_users: %d" % user_stat.count()
    #     user_stat = user_stat.filter(lambda x: x[-1] == 0)
    #     print "no response users: %d" % user_stat.count()
    #     data = data.map(lambda x: (int(x["user_id"]), x)).repartition(40)\
    #         .subtractByKey(user_stat).map(lambda (uid, x): x)
    #     print data.count()
    df = spark.createDataFrame(data.map(mapper2).filter(lambda x: x is not None), schema=schema)
    df.repartition(10).write.format("tfrecords").mode("overwrite").save(
        "/tmp/research/mall_%s_%s_%d/" % (model_type, start_dt, days))


if __name__ == "__main__":
    spark, sc = init_spark("omall_click_dataset")
    model_type = "click_model_v2"
    sc.setLogLevel("ERROR")
    cur_ts = sc.broadcast(int(time.time()))
    d1 = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    d2 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    train_dt = sys.argv[3] if len(sys.argv) > 3 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400 * (1 + d2)))
    get_dataset(train_dt, d1)
    test_dt = sys.argv[4] if len(sys.argv) > 4 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    get_dataset(test_dt, d2)
