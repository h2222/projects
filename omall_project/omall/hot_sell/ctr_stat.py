# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import sys
import time
from operator import add
import os
alter_command = """
curl 'https://oapi.dingtalk.com/robot/send?access_token=f01c80b52d4bb8f5d637a98a82c6276ce6e4a424a29bb98c988ef53a56d01047' \
-H 'Content-Type: application/json' \
-d '
{
    "msgtype": "text", 
    "text": {
        "content": "warn: %s"
    }
}'
"""

# show & click & into
query_opay_client = """
select message from oride_source.opay_ep_logv1 where dt='{dt}' and
instr(message, "TAB_MALL") > 0
"""


def opay_client_mapper(x):
    try:
        ret = []
        tmp = json.loads(x.message)
        raw_uid = tmp.get("uid", '')
        ts = int(float(tmp["t"]) / 1000.0)
        events = tmp["es"]
        for e in events:
            uid = 1
            raw_cur_uid = e.get("uid", '')
            if raw_uid.isdigit():
                uid = int(raw_uid)
            elif raw_cur_uid.isdigit():
                uid = int(raw_cur_uid)
            event_name = e['en'].strip()
            val = e['ev']
            if event_name == "TAB_MALL_homepage_hotproducts_commoditylist_clik":
                gid = val.get('value', '')
                if gid.isdigit():
                    ret.append((uid, int(gid), ts, 1))
            elif event_name == "TAB_MALL_hot_product_show":
                gid = val.get('value', '')
                if gid.isdigit():
                    ret.append((uid, int(gid), ts, 0))
        return ret
    except:
        pass


def percent_format(x):
    return "{:.2%}".format(x)


def get_ctr_stat(dt, d1, d2):
    default_ctr_data_show, default_ctr_data_click = d1.get(0, 0), d1.get(1, 0)
    user_ctr_data_show, user_ctr_data_click = d2.get(0, 0), d2.get(1, 0)
    default_user_ctr = percent_format(
        default_ctr_data_click / float(default_ctr_data_show) if default_ctr_data_show > 0 else 0)
    user_ctr = percent_format(user_ctr_data_click / float(user_ctr_data_show) if user_ctr_data_show > 0 else 0)
    print "========================="
    s1 = "dt: %s, compare group: show: %d, click: %d, ctr: %s" \
          % (dt, default_ctr_data_show, default_ctr_data_click, default_user_ctr)
    s2 = "dt: %s, test group: show: %d, click: %d, ctr: %s" % (dt, user_ctr_data_show, user_ctr_data_click, user_ctr)
    print s1
    print s2
    os.system(alter_command % "\n".join(["", "B2C热销推荐位置", s1, s2]))


def hot_sell_ctr_stat(dt):
    data = spark.sql(query_opay_client.format(dt=dt)) \
        .rdd.map(opay_client_mapper).filter(lambda x: x is not None).flatMap(lambda x: x).distinct().cache()
    default_ctr_data = dict(data.filter(lambda x: x[0] % 10 == 0).map(lambda x: (x[-1], 1)).reduceByKey(add).collect())
    user_ctr_data = dict(data.filter(lambda x: x[0] % 10 != 0 and x[0] != 1).map(lambda x: (x[-1], 1)).reduceByKey(add).collect())
    get_ctr_stat(dt, default_ctr_data, user_ctr_data)


if __name__ == "__main__":
    spark, sc = init_spark("omall_daily")
    sc.setLogLevel("ERROR")
    day_range = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i in range(day_range):
        day = time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400 * (i + 1)))
        print day
        hot_sell_ctr_stat(day)
