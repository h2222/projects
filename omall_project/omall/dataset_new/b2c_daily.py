# -*- coding: utf-8 -*-
import time
from init_spark import init_spark
import json
import sys
from consts import *


def feature_mapper(x):
    try:
        tmp_dict = x.asDict()
        good_id = int(tmp_dict["good_id"])
        user_id = int(tmp_dict["user_id"])
        ts = int(tmp_dict["timestamp"])
        basic_data = map(int, [tmp_dict.get(x, "0") if tmp_dict.get(x, "0") is not None else "0" for x in
                               sc_basic_c.value])
        int_stat_data = map(float, [tmp_dict.get(x, "0") if tmp_dict.get(x, "0") is not None else "0" for x in
                                    sc_int_stat_basic_columns.value])
        float_data = map(float, [tmp_dict.get(x, "0") if tmp_dict.get(x, "0") is not None else "0" for x in
                                 sc_float_basic_columns.value])
        str_data = map(str, [tmp_dict.get(x, "") if tmp_dict.get(x, "") is not None else "" for x in
                             sc_str_basic_columns.value])
        score = float(tmp_dict.get("score", 0)) if tmp_dict.get("score", 0) is not None else 0.0
        recall_source = tmp_dict.get("recall_source", []) if tmp_dict.get("recall_source",
                                                                                 []) is not None else []
        return (user_id, good_id), (ts, basic_data, int_stat_data, float_data, str_data, score, recall_source)
    except:
        pass


def get_feature_data(last_day):
    for i in range(24):
        str_h = str(i)
        str_h = "0" + str_h if len(str_h) == 1 else str_h
        print last_day, str_h
        spark.sql(
            "ALTER TABLE algo.b2c_mall_feature ADD IF NOT EXISTS PARTITION (dt = '%s', hour = '%s')" % (
                last_day, str_h))
    data = spark.sql("""
    select * from algo.b2c_mall_feature where dt="{dt}" and user_id != 1
    """.format(dt=last_day)).rdd.map(feature_mapper).filter(lambda x: x is not None)
    return data


def get_label_data(last_day):
    data = sc.textFile("oss://opay-datalake/algo_migration/omall/event_label/dt=%s" % last_day) \
        .map(json.loads).filter(lambda x: x is not None and int(x["user_id"]) != 1) \
        .map(lambda x: ((int(x["user_id"]), int(x["good_id"])), (int(x["ts"]), int(x["label"]))))
    return data


def merge_mapper(x):
    try:
        ((user_id, good_id), stat_all) = x
        if len(stat_all) > 0:
            ret = []
            stat_all.sort(key=lambda i: i[0])
            for i in range(len(stat_all) - 1):
                if len(stat_all[i]) == 7:
                    for j in range(i + 1, len(stat_all)):
                        if len(stat_all[j]) == 2:
                            (f_ts, basic_data, int_stat_data, float_data, str_data, score, recall_source) = stat_all[i]
                            (l_ts, label) = stat_all[j]
                            ret.append(
                                (user_id, good_id, l_ts, label, basic_data, int_stat_data, float_data, str_data, score,
                                 recall_source))
                        else:
                            break
            if len(ret) > 0:
                return ret
    finally:
        pass


def percent_format(x):
    return "{:.2%}".format(x)


def get_daily_data(dt):
    label_data = get_label_data(dt)
    feature_data = get_feature_data(dt)
    data = label_data.union(feature_data).repartition(50).groupByKey().mapValues(list)
    data = data.map(merge_mapper).filter(lambda x: x is not None) \
        .flatMap(lambda x: x) \
        .map(lambda (user_id, good_id, l_ts, label, basic_data, int_stat_data,
                    float_data, str_data, score, recall_source):
             json.dumps({
                 "user_id": user_id,
                 "good_id": good_id,
                 "ts": l_ts,
                 "label": label,
                 "basic_data": basic_data,
                 "stat_data": int_stat_data,
                 "float_data": float_data,
                 "str_data": str_data,
                 "score": score,
                 "recall_source": recall_source,
             })).filter(lambda x: x is not None)
    cnt = data.count()
    print cnt
    if cnt > 0:
        data.repartition(10).saveAsTextFile(dataset_path.format(model_type=model_type, dt=dt),
                                            compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":
    model_type = "click_model_v2"
    spark, sc = init_spark(model_type)
    sc.setLogLevel("WARN")
    sc_basic_c = sc.broadcast(basic_columns)
    sc_int_stat_basic_columns = sc.broadcast(int_stat_basic_columns)
    sc_float_basic_columns = sc.broadcast(float_basic_columns)
    sc_str_basic_columns = sc.broadcast(str_basic_columns)
    dataset_path = "oss://opay-datalake/algo_migration/omall/{model_type}/dt={dt}"
    train_dt = str(sys.argv[1]) if len(sys.argv) > 1 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    print train_dt, model_type
    get_daily_data(train_dt)
