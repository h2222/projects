# -*- coding: utf-8 -*-
import redis
import time
import os
import json
from consts import *
from util import *

# prod env
rds_host = "r-d7oven45vpwhoe42ev.redis.eu-west-1.rds.aliyuncs.com"
rds_port = 6379

# test env
# rds_host = "10.52.176.96"
# rds_port = 6379


expire_time = 86400 * 7
pipe_command_num = 256
rds = redis.StrictRedis(host=rds_host, port=rds_port)


def mall_rec2rds():
    # 1.download_via_oss
    bucket = get_bucket()
    file_path = "/data/omall_rec.txt"
    if os.path.exists(file_path):
        os.system("rm -rf %s" % file_path)
    remote_path = "omall_rec/goods.txt"
    bucket.get_object_to_file(remote_path, file_path)
    f = open(file_path)
    for line in f:
        try:
            tmp = json.loads(line.strip())
            algo = tmp["algo"]
            mtype = tmp["type"]
            goods_id = tmp["goods_id"]
            raw_id = 0
            if mtype == uid2goods:
                raw_id = tmp["user_id"]
            elif mtype == goods2goods:
                raw_id = tmp["item_id"]
            if raw_id != 0:
                key = b2c_algo_recall_key % (mtype, algo, raw_id)
                rds.set(key, json.dumps({
                    "good_ids": goods_id
                }), ex=expire_time)
                time.sleep(0.05)
        except:
            pass


if __name__ == "__main__":
    mall_rec2rds()
