# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import time
import sys
reload(sys)
sys.setdefaultencoding('utf8')

query_opay_client = """
select message from oride_source.opay_ep_logv1 where dt = '{dt}'
and instr(message, "TAB_MALL_Searchresultpage") > 0
"""


def remove_digit(x):
    res = ""
    for e in x:
        if not e.isdigit():
            res += e
    return res


def split_digit(x):
    res = ""
    x_len = len(x)
    if x_len > 2:
        for i in range(x_len - 1):
            res += x[i]
            cur_is_digit = x[i].isdigit()
            next_is_digit = x[i + 1].isdigit()
            if cur_is_digit != next_is_digit:
                res += " "
        res += x[x_len - 1]
    else:
        res = x
    return res


def filter_special_str(x):
    res = ""
    for s in x:
        if s not in (",", "(", ")", '"', "/", ":", "&", "-",
                     '\\', "{", "}", "+", "【", "】", "*", ";", '.', "<",
                     ">", "=", '[', "]", "|", "_", "%", "$", "#", "!", "@", "^"):
            res += s
        elif s == "'":
            res += ""
        else:
            res += " "
    return res


def none_str(x):
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    return str(x).strip().lower() if x is not None else ""


def pure_str(x):
    if x == "":
        return x
    return filter_special_str(split_digit(none_str(x)))


def opay_client_mapper(x):
    try:
        ret = []
        tmp = json.loads(x.message)
        events = tmp["es"]
        for e in events:
            event_name = e['en'].strip()
            val = e['ev']
            raw_key_word = val.get('keyword', '')
            keyword = pure_str(raw_key_word)
            if keyword != "":
                if event_name == "TAB_MALL_Searchresultpage_haveresults_show":
                    ret.append(("y", keyword))
                elif event_name == "TAB_MALL_Searchresultpage_noresults_show":
                    ret.append(("n", keyword))
        if len(ret) > 0:
            return ret
    except:
        pass


def search_res(dt):
    spark.sql(query_opay_client.format(dt=dt)) \
        .rdd.map(opay_client_mapper).filter(lambda x: x is not None).flatMap(lambda x: x)\
        .distinct() \
        .map(lambda (res_type, keyword): json.dumps({
            "st": res_type,
            "keyword": keyword,
        })).filter(lambda x: x is not None).repartition(1) \
        .saveAsTextFile("oss://opay-datalake/algo_migration/omall/b2c_search/dt=%s" % dt,
                        compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


if __name__ == "__main__":
    spark, sc = init_spark("b2c_search")
    sc.setLogLevel("ERROR")
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    print day
    search_res(day)
