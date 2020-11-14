# -*- coding: utf-8 -*-
from sklearn.metrics import roc_auc_score
import os
from init_spark import init_spark
import json
import sys
import time

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


def get_user_group(ucode):
    uc = ucode % 10
    if uc == 0:
        return 0
    elif uc == 1:
        return 1
    else:
        return 2


# def get_recall_user_group(ucode):
#     uc = ucode % 10
#     pass


def percent_format(x):
    return "{:.2%}".format(x)


def mapper(x):
    try:
        return int(x["user_id"]), x["label"], x["score"], x.get("recall_source", [])
    except:
        pass


def b2c_auc_check(dt):
    data = sc.textFile(dataset_path.format(model_type=model_type, dt=dt)).map(json.loads).filter(
        lambda x: x is not None).map(mapper).filter(
        lambda x: x is not None).collect()
    l = [x[1] for x in data]
    s = [x[2] for x in data]
    dt_auc = percent_format(roc_auc_score(l, s))
    print "dt: %s, auc: %s" % (dt, dt_auc)
    # os.system(alter_command % ("Date %s | modelType: %s | online auc: %s" % (dt, model_type, dt_auc)))
    user_group_check = {}
    recall_source_check = {}
    for elem in data:
        (user_group, label, score, recall_source) = elem
        uc = get_user_group(user_group)
        if uc not in user_group_check:
            user_group_check[uc] = {}
            user_group_check[uc]["label"] = []
            user_group_check[uc]["score"] = []
        user_group_check[uc]["label"].append(label)
        user_group_check[uc]["score"].append(score)
        if not recall_source:
            rs = "null"
            if rs not in recall_source_check:
                recall_source_check[rs] = {}
                recall_source_check[rs]["label"] = []
                recall_source_check[rs]["score"] = []
            recall_source_check[rs]["label"].append(label)
            recall_source_check[rs]["score"].append(score)
        else:
            for rs in recall_source:
                if rs not in recall_source_check:
                    recall_source_check[rs] = {}
                    recall_source_check[rs]["label"] = []
                    recall_source_check[rs]["score"] = []
                recall_source_check[rs]["label"].append(label)
                recall_source_check[rs]["score"].append(score)
    for ug in user_group_check:
        if len(user_group_check[ug]) > 0 and len(user_group_check[ug]["label"]) > 0:
            try:
                cur_auc = percent_format(roc_auc_score(user_group_check[ug]["label"], user_group_check[ug]["score"]))
                print "dt: %s, user group : %d,  related num: %d, auc: %s" % \
                      (dt, ug, len(user_group_check[ug]["label"]), cur_auc)
            except Exception as e:
                print e.message
                pass
    for rs in recall_source_check:
        if len(recall_source_check[rs]) > 0 and len(recall_source_check[rs]["label"]) > 0:
            try:
                cur_auc = percent_format(roc_auc_score(recall_source_check[rs]["label"], recall_source_check[rs]["score"]))
                print "dt: %s, recall source: %s,  related num: %d, auc: %s" % \
                      (dt, rs, len(recall_source_check[rs]["label"]), cur_auc)
            except Exception as e:
                print e.message
                pass


if __name__ == "__main__":
    model_type = "click_model_v2"
    spark, sc = init_spark("b2c_auc_check")
    sc.setLogLevel("ERROR")
    day = str(sys.argv[1]) if len(sys.argv) > 1 else \
        time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    print day
    dataset_path = "oss://opay-datalake/algo_migration/omall/{model_type}/dt={dt}"
    b2c_auc_check(day)
