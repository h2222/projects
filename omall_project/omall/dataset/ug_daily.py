# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import sys
import time
from consts import *

# only with payment
order_query = """
select opayid, goods_id, pay_time, __ts_ms / 1000 as ts from
(select id, opayid, pay_time, __ts_ms from otrade_dw_ods.ods_binlog_mall_nideshop_order_hi
where dt = '{dt}') as t1
join
(select order_id, goods_id from otrade_dw_ods.ods_binlog_mall_nideshop_order_goods_hi
where dt ='{dt}') as t2
on t1.id=t2.order_id
"""


def get_pair(dt):
    data = spark.sql(order_query.format(dt=dt)).rdd.cache()
    res = data.map(lambda x: json.dumps({
        "user_id": int(x.opayid),
        "good_id": int(x.goods_id),
        "ts": int(x.ts),
        "label": 1 if x.pay_time is not None and int(x.pay_time) > 0 else 0
    })).filter(lambda x: x is not None)
    print res.count()
    res.repartition(1).saveAsTextFile(
        "oss://opay-datalake/algo_migration/omall/og_pair/dt=%s" % dt)


# show & click & into
query_opay_client = """
select message from oride_source.opay_ep_logv1 where dt='{dt}' and
instr(message, "TAB_MALL") > 0
"""

"""
TAB_MALL_detailspage_share_clik
{u'en': u'TAB_MALL_detailspage_share_clik', u'uid': u'156619082756750700', u'bzp': u'TAB_MALL', u'cid': u'ngLagos', u'cip': u'fe80::b4c9:20ff:fe38:d6f7%dummy0', u'sid': u'', u'lat': u'6.5924943', u'et': u'1592125688419', u'lng': u'3.3585913', u'ev': {u'spu_id': u'1183824', u'sku_id': u'4128'}, u'uno': u'8039275922'}

TAB_MALL_detailspage_cart_clik
{u'en': u'TAB_MALL_detailspage_cart_clik', u'uid': u'156620021727026268', u'bzp': u'TAB_MALL', u'cid': u'', u'cip': u'10.178.164.137', u'sid': u'p_239509312', u'lat': u'5.9425912', u'et': u'1592129763548', u'lng': u'5.6630502', u'ev': {}, u'uno': u'8122933506'}

TAB_MALL_detailspage_addtocart_clik
{u'en': u'TAB_MALL_detailspage_addtocart_clik', u'uid': u'156619111922643073', u'bzp': u'TAB_MALL', u'cid': u'', u'cip': u'100.78.47.127', u'sid': u'', u'lat': u'null', u'et': u'1592089984703', u'lng': u'null', u'ev': {u'spu_id': u'1183814', u'sku_id': u'4115'}, u'uno': u'7032058607'}

TAB_MALL_detailspage_select_clik
{u'en': u'TAB_MALL_detailspage_select_clik', u'uid': u'156619082753164900', u'bzp': u'TAB_MALL', u'cid': u'ngLagos', u'cip': u'10.244.8.155', u'sid': u'', u'lat': u'6.4550884', u'et': u'1592103796703', u'lng': u'3.3563712', u'ev': {u'spu_id': u'1184254'}, u'uno': u'8079127785'}
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
            elif event_name in sc_candidate_event_name.value:
                gid = val.get('spu_id', '')
                if gid.isdigit():
                    ret.append((uid, int(gid), ts, 1))
        return ret
    except:
        pass


def reducer(a, b):
    res = [0] * len(a)
    for x in range(len(a)):
        res[x] = a[x] + b[x]
    return res


def opay_data_daily(dt):
    data = spark.sql(query_opay_client.format(dt=dt)) \
        .rdd.map(opay_client_mapper).filter(lambda x: x is not None).flatMap(lambda x: x).distinct().cache()
    label = data.map(lambda (uid, gid, ts, label): ((uid, gid), (label, ts))) \
        .map(lambda ((uid, gid), (label, ts)): json.dumps({
            "user_id": uid,
            "good_id": gid,
            "label": label,
            "ts": ts,
        })).filter(lambda x: x is not None)
    print label.count()
    label.repartition(1) \
        .saveAsTextFile("oss://opay-datalake/algo_migration/omall/event_label/dt=%s" % dt,
                        compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")
    res = data.map(lambda (uid, gid, ts, label): ((uid, gid), [1, 0] if label > 0 else [0, 1])) \
        .reduceByKey(reducer) \
        .map(lambda ((uid, gid), (click_val, show_val)): json.dumps({
            "user_id": uid,
            "good_id": gid,
            "click_num": click_val,
            "show_num": show_val,
        })).filter(lambda x: x is not None)
    print res.count()
    res.repartition(1).saveAsTextFile("oss://opay-datalake/algo_migration/omall/opg_pair/dt=%s" % dt,
                                      compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":
    spark, sc = init_spark("omall_daily")
    sc.setLogLevel("ERROR")
    sc_candidate_event_name = sc.broadcast(candidate_event_name)
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    print day
    get_pair(day)
    opay_data_daily(day)
