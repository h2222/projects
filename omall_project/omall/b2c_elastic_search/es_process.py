# -*- coding: utf-8 -*-
import time
import sys
import elasticsearch
from elasticsearch.helpers import bulk
import json
import oss2
import os
reload(sys)
sys.setdefaultencoding('utf8')

"""
algo-tensorflow-01-8.208.16.172-10.52.28.207
  
algo-tensorflow-02-8.208.16.2-10.52.28.209
"""
ELASTICSEARCH_HOST = ["http://127.0.0.1:9200"]

root_index = "b2c_algo_es"
dt_index = "b2c_algo_es_{dt}"


def build_root_index(index_name, params):
    if not es.indices.exists(index=index_name):
        es.indices.create(index_name, body=params)


def build_index(index_name, params):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index_name, body=params)


def update_index_alias(index_name, alias_index_name):
    actions = []
    histroy_index = []
    response = es.indices.exists_alias(name=alias_index_name)
    if response:
        response = es.indices.get_alias(name=alias_index_name)
        for key in response:
            histroy_index.append(key)
            action = {
                "remove": {
                    "index": key,
                    "alias": alias_index_name
                }
            }
            actions.append(action)
    else:
        es.indices.put_alias(index=index_name, name=alias_index_name)
        return
    action = {
        "add": {
            "index": index_name,
            "alias": alias_index_name
        }
    }
    actions.append(action)
    es.indices.update_aliases({"actions": actions})
    if len(histroy_index) < 1:
        return
    for key in histroy_index:
        res = es.indices.exists(index=key)
        if res:
            es.indices.delete(index=key)


GOOD_NAME_PARAM = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "good_name": {
                "type": "text",
                "analyzer": "icu_analyzer",
                "search_analyzer": "icu_analyzer",
            },
            "good_id": {"type": "long"},
        }
    }
}

goods_query = """
select id, update_time, name from
otrade_dw.dwd_otrade_b2c_mall_nideshop_goods_hf where dt = '{dt}' and is_delete=0 and is_on_sale=1 
"""


def get_bucket():
    auth = oss2.Auth('LTAI4FmshDAuao6Wyn1BAoGB', 'MLhLZpdpL1sNrtUNsfcfcFj4eHq7kp')
    bucket = oss2.Bucket(auth, 'http://oss-eu-west-1-internal.aliyuncs.com', 'oride-algo')
    return bucket


def get_good_names():
    bucket = get_bucket()
    file_path = "/data/good_name_es.txt"
    if os.path.exists(file_path):
        os.system("rm -rf %s" % file_path)
    remote_path = "omall_b2c_es/good_names.txt"
    bucket.get_object_to_file(remote_path, file_path)
    f = open(file_path)
    good_data = {}
    for line in f:
        try:
            tmp = json.loads(line.strip())
            good_data[int(tmp["good_id"])] = tmp["good_name"]
        except Exception as e:
            print e.message
            pass
    os.system("rm -rf %s" % file_path)
    return good_data


def bulk_good_name_data(index_name):
    good_data = get_good_names()
    arr = []
    for good_id in good_data:
        name = good_data[good_id]
        name = str(name.encode('utf-8'))
        act = {
            '_op_type': 'create',
            '_index': index_name,
            '_id': good_id,
            '_source': {
                'good_name': name,
                'good_id': good_id,
            }
        }
        arr.append(act)
    print len(arr)
    bulk(es, arr, True, raise_on_error=False, raise_on_exception=False)
    print "done"


if __name__ == "__main__":
    es = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y%m%d', time.localtime(time.time() - 86400))
    cur_index = dt_index.format(dt=day)
    print day, cur_index
    # build_root_index(root_index, GOOD_NAME_PARAM)
    build_index(cur_index, GOOD_NAME_PARAM)
    bulk_good_name_data(cur_index)
    update_index_alias(cur_index, root_index)
