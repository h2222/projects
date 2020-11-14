# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import time
from operator import add

# show & click & into
query_opay_client = """
select message from oride_source.opay_ep_logv1 where dt >= '{dt}' and
instr(message, "MALL") > 0
"""


def mapper(x):
    try:
        ret = []
        tmp = json.loads(x.message)
        events = tmp["es"]
        for e in events:
            event_name = e['en'].strip()
            ret.append(event_name)
        return ret
    except:
        pass


def event_check(dt):
    data = spark.sql(query_opay_client.format(dt=dt))\
        .rdd.map(mapper).filter(lambda x: x is not None)\
        .flatMap(lambda x: x)\
        .map(lambda x: (x, 1)).reduceByKey(add)\
        .repartition(1).sortBy(lambda x: x[-1], ascending=False).collect()
    for elem in data:
        print elem


if __name__ == "__main__":
    spark, sc = init_spark("opay_event_check")
    sc.setLogLevel("ERROR")
    yt = time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    print yt
    event_check(yt)
