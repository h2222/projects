# -*- coding: utf-8 -*-
import oss2
from init_spark import init_spark
import time
import sys
import os
import json


def get_bucket():
    auth = oss2.Auth('LTAI4FmshDAuao6Wyn1BAoGB', 'MLhLZpdpL1sNrtUNsfcfcFj4eHq7kp')
    bucket = oss2.Bucket(auth, 'http://oss-eu-west-1-internal.aliyuncs.com', 'oride-algo')
    return bucket


def upload_via_oss(fp, oss_path):
    bucket = get_bucket()
    with open(fp, 'rb') as fileobj:
        bucket.put_object(oss_path, fileobj)


goods_query = """
select id, update_time, name from
otrade_dw.dwd_otrade_b2c_mall_nideshop_goods_hf where dt = '{dt}' and is_delete=0 and is_on_sale=1 
"""


def none_str(x):
    return str(x) if x is not None else ""


def goods_mapper(x):
    try:
        # keywords SD card, Memory card
        return int(x.id), (str(x.update_time), none_str(x.name))
    except:
        pass


def get_good_names(dt):
    return dict(spark.sql(goods_query.format(dt=dt)).rdd.map(goods_mapper) \
                .filter(lambda x: x is not None).reduceByKey(max) \
                .map(lambda (gid, (_, name)): (gid, name)).collect())


def get_fp(f_path):
    if os.path.exists(f_path):
        os.system("rm -rf %s" % f_path)
    return open(f_path, "aw")


def upload_good_name_data(dt):
    file_path = "/data/omall_b2c_nlp/b2c_es_%s.txt" % dt
    f = get_fp(file_path)
    good_data = get_good_names(dt)
    for good_id in good_data:
        try:
            name = good_data[good_id]
            name = str(name.encode('utf-8'))
            f.write(json.dumps({
                'good_name': name,
                'good_id': good_id,
            }))
            f.write("\n")
        except Exception as e:
            print e.message
            pass
    upload_via_oss(file_path, "omall_b2c_es/good_names.txt")


if __name__ == "__main__":
    spark, sc = init_spark("omall_elastic_search")
    sc.setLogLevel("ERROR")
    cur_ts = sc.broadcast(int(time.time()))
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    upload_good_name_data(day)