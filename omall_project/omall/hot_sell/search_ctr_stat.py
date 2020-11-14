# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import sys
import time
from operator import add

"""
(u'TAB_MALL_Searchresultpage_haveresults_show', 5402) 搜索结果页展示-有搜索结果
[{u'en': u'TAB_MALL_Searchresultpage_haveresults_show', u'uid': u'156619110158324648', u'bzp': u'TAB_MALL', u'cid': u'ngLagos', u'cip': u'100.67.82.7', u'sid': u'', u'lat': u'6.6157024', u'et': u'1592094861502', u'lng': u'3.3160366', u'ev': {u'category_id': u'6916', u'keyword': u''}, u'uno': u'9033443150'}]


(u'TAB_MALL_Searchresultpage_noresults_show', 1255) 搜索结果页展示-无搜索结果
{u'en': u'TAB_MALL_Searchresultpage_noresults_show', u'uid': u'156619082746754400', u'bzp': u'TAB_MALL', u'cid': u'', u'cip': u'10.80.25.5', u'sid': u'p_235519882', u'lat': u'6.6402977', u'et': u'1592129581669', u'lng': u'3.3353322', u'ev': {u'category_id': u'', u'keyword': u'ifinix'}, u'uno': u'8188267800'}


(u'TAB_MALL_homepage_Searchbox_clik', 1134)
{u'en': u'TAB_MALL_homepage_Searchbox_clik', u'uid': u'156620061084746518', u'bzp': u'TAB_MALL', u'cid': u'', u'cip': u'10.20.148.138', u'sid': u'', u'lat': u'7.3803537', u'et': u'1592098611174', u'lng': u'3.8195162', u'ev': {}, u'uno': u'8136093357'}


(u'TAB_MALL_Searchresultpage_sales_click', 333)   在搜索结果页点击销量排序
{u'en': u'TAB_MALL_Searchresultpage_sales_click', u'uid': u'156620050567956411', u'bzp': u'TAB_MALL', u'cid': u'', u'cip': u'10.4.170.161', u'sid': u'', u'lat': u'7.2727369', u'et': u'1592168266497', u'lng': u'5.1729414', u'ev': {}, u'uno': u'8140345661'}


(u'TAB_MALL_searchpage_hot_click', 85)
{u'en': u'TAB_MALL_searchpage_hot_click', u'uid': u'156619122214262001', u'bzp': u'TAB_MALL', u'cid': u'ngKaduna', u'cip': u'10.202.66.222', u'sid': u'', u'lat': u'null', u'et': u'1592101365073', u'lng': u'null', u'ev': {u'name': u'zealot', u'value': u'6'}, u'uno': u'7067790107'}

"""

# show & click & into
query_opay_client = """
select message from oride_source.opay_ep_logv1 where dt='{dt}' and
instr(message, "TAB_MALL") > 0
"""

search_event_arr = [
    "TAB_MALL_Searchresultpage_haveresults_show",
    "TAB_MALL_Searchresultpage_noresults_show",
    "TAB_MALL_homepage_Searchbox_clik",
    "TAB_MALL_Searchresultpage_sales_click",
    "TAB_MALL_searchpage_hot_click",
]


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
            if event_name in sc_search_event_arr.value > 0:
                ret.append(((uid, event_name), 1))
        if len(ret) > 0:
            return ret
    except:
        pass


def percent_format(x):
    return "{:.2%}".format(x)


def get_search_stat(d1, d2):
    for key in d1:
        print "default users: ", key, d1[key]
    print "========******************========"
    for key in d2:
        print "users: ", key, d2[key]
    default_search_haveres, default_search_nores = \
        d1.get("TAB_MALL_Searchresultpage_haveresults_show", 0), \
        d1.get("TAB_MALL_Searchresultpage_noresults_show", 0)
    user_search_haveres, user_search_nores = \
        d2.get("TAB_MALL_Searchresultpage_haveresults_show", 0), \
        d2.get("TAB_MALL_Searchresultpage_noresults_show", 0)
    default_user_haveres_rate = percent_format(
        default_search_haveres / float(default_search_haveres + default_search_nores))
    user_haveres_rate = percent_format(user_search_haveres / float(user_search_haveres + user_search_nores))
    print "group1 default user haveres: %d, default user nores: %d, default user haveres rate: %s" \
          % (default_search_haveres, default_search_nores, default_user_haveres_rate)
    print "group2 user haveres: %d, user nores: %d, user haveres rate: %s" \
          % (user_search_haveres, user_search_nores, user_haveres_rate)


def hot_sell_search_stat(dt):
    data = spark.sql(query_opay_client.format(dt=dt)) \
        .rdd.map(opay_client_mapper).filter(lambda x: x is not None).flatMap(lambda x: x).cache()
    default_data = dict(
        data.filter(lambda (x, y): x[0] == 1).map(lambda (x, y): (x[1], 1)).reduceByKey(add).collect())
    user_data = dict(data.filter(lambda (x, y): x[0] != 1).map(lambda (x, y): (x[1], 1)).reduceByKey(add).collect())
    get_search_stat(default_data, user_data)
    group1_data = dict(
        data.filter(lambda (x, y): x[0] % 10 not in [8, 9] and x[0] != 1).map(lambda (x, y): (x[1], 1)).reduceByKey(
            add).collect())
    group2_data = dict(
        data.filter(lambda (x, y): x[0] % 10 in [8, 9]).map(lambda (x, y): (x[1], 1)).reduceByKey(add).collect())
    get_search_stat(group1_data, group2_data)


if __name__ == "__main__":
    spark, sc = init_spark("omall_daily")
    sc.setLogLevel("ERROR")
    sc_search_event_arr = sc.broadcast(search_event_arr)
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    print day
    hot_sell_search_stat(day)
