# -*- coding: utf-8 -*-
import os
import oss2
import numpy as np
from consts import *
import redis
import json
import time
import random


def download_data(dt, type_name):
    root_path = "/data/omall/%s" % type_name
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    tmp_dataset_path = "/tmp/omall/%s/dt=%s/" % (type_name, dt)
    os.system("/usr/bin/sh /data/download.sh %s %s" % (tmp_dataset_path, root_path))


def remove_data(dt, type_name):
    path1 = "/data/omall/%s/dt=%s/" % (type_name, dt)
    os.system("rm -rf %s" % path1)
    path1 = "/tmp/omall/%s/dt=%s/" % (type_name, dt)
    os.system("/usr/lib/hadoop-current/bin/hdfs dfs -rm -r -skipTrash %s" % path1)


def get_bucket():
    auth = oss2.Auth('LTAI4FmshDAuao6Wyn1BAoGB', 'MLhLZpdpL1sNrtUNsfcfcFj4eHq7kp')
    bucket = oss2.Bucket(auth, 'http://oss-eu-west-1-internal.aliyuncs.com', 'oride-algo')
    return bucket


def upload_via_oss(fp, oss_path):
    bucket = get_bucket()
    with open(fp, 'rb') as fileobj:
        bucket.put_object(oss_path, fileobj)
    # os.system("rm -rf %s" % fp)


def random_shuffle(x):
    return round((1 + (random.random() - 0.5) * 0.3) * x, 2)


def get_personas_goods(m1, m2, d1, d2, valid_ids=None, max_goods_num=20):
    rd1 = {}
    rd2 = {}
    for d in d1:
        rd1[d1[d]] = d
    for d in d2:
        rd2[d2[d]] = d
    score_matrix = np.matmul(m1, m2.transpose())
    row, column = score_matrix.shape[0], score_matrix.shape[1]
    column_idx = range(column)
    recommend_res = {}
    repeated_stat = {}
    for i in range(len(rd1)):
        try:
            id1 = rd1[i]
            cur_scores = map(random_shuffle, list(score_matrix[i, :]))
            item_scores = zip(column_idx, cur_scores)
            item_scores.sort(key=lambda x: x[-1], reverse=True)
            rec_items = [rd2[x[0]] for x in item_scores[:min(len(item_scores), max_goods_num)] if rd2[x[0]] != id1]
            if valid_ids is not None and len(valid_ids) > 0:
                recommend_res[id1] = list(set(rec_items) & set(valid_ids))
            else:
                recommend_res[id1] = rec_items
            for item in recommend_res[id1]:
                if item not in repeated_stat:
                    repeated_stat[item] = 0
                repeated_stat[item] += 1
        except Exception as e:
            print e.message
            pass
    repeated_data = zip(repeated_stat.keys(), repeated_stat.values())
    print "推荐item覆盖率  {:.2%}".format(len(repeated_data) / float(column) if column > 0 else 0)
    repeated_data.sort(key=lambda x: x[-1], reverse=True)
    for e in repeated_data:
        print e
    return recommend_res


def get_default_goods(algo_type, key_type, m1, m2, d2, valid_ids=None, max_goods_num=20):
    rd2 = {}
    for d in d2:
        rd2[d2[d]] = d
    m1_r, m1_c = m1.shape
    m = np.mean(m1, axis=0).reshape((1, m1_c))
    score_matrix = np.matmul(m, m2.transpose())
    row, column = score_matrix.shape[0], score_matrix.shape[1]
    column_idx = range(column)
    cur_score = list(score_matrix[0, :])
    item_scores = zip(column_idx, cur_score)
    item_scores.sort(key=lambda x: x[-1], reverse=True)
    rec_items = [rd2[x[0]] for x in item_scores[:min(len(item_scores), max_goods_num)]]
    if valid_ids is not None and len(valid_ids) > 0:
        rec_items = list(set(rec_items) & set(valid_ids))
    if algo_type == "" or len(rec_items) == 0 or (key_type not in ["u2g", "g2g"]):
        return
    rds = redis.StrictRedis(host=rds_host, port=rds_port)
    key = b2c_algo_recall_key % (key_type, algo_type, default_user_id)
    print key
    rds.set(key, json.dumps({
        "good_ids": rec_items
    }), ex=expire_time)


def write_algo2rds(res, algo_type, key_type):
    if algo_type == "":
        return
    if key_type not in ["u2g", "g2g"]:
        return
    rds = redis.StrictRedis(host=rds_host, port=rds_port)
    rpipe = rds.pipeline(transaction=False)
    #
    pipe_command_num = 64
    counter = 0
    for i in res:
        key = b2c_algo_recall_key % (key_type, algo_type, i)
        print key, len(res[i])
        rpipe.set(key, json.dumps({
            "good_ids": res[i]
        }), ex=expire_time)
        counter += 1
        if counter > 0 and counter % pipe_command_num == 0:
            time.sleep(0.5)
            rpipe.execute()
    rpipe.execute()
    rpipe.close()


def write2localfile(f_path, algo, data, data_type):
    id_str = "user_id"
    if data_type == "g2g":
        id_str = "item_id"
    for rid in data:
        f_path.write(json.dumps({
            "algo": algo,
            "type": data_type,
            id_str: rid,
            "goods_id": data[rid]
        }))
        f_path.write("\n")


def get_nlp_data(nlp_dt):
    good2name_words_dict = {}
    nlp_data_path = "/data/omall_b2c_nlp/goods_nlp_%s.txt" % nlp_dt
    if not os.path.exists(nlp_data_path):
        return good2name_words_dict
    """
    "good_id": id,
    "name_words": id_name_words,
    "keyword_words": id_keywords_words,
    "desc_words": id_desc_words
    """
    raw_dict = json.loads(open(nlp_data_path).readline())
    for gid in raw_dict:
        try:
            name_words = raw_dict[gid]['name_words'] + raw_dict[gid]['brand_words'] + raw_dict[gid]['category_words']
            gid = int(gid)
            if gid not in good2name_words_dict:
                good2name_words_dict[gid] = set()
            good2name_words_dict[gid].update(name_words)
        except Exception as e:
            print e.message
            pass
    return good2name_words_dict


def scalarization(data):
    data = np.array(data)
    _range = np.max(data) - np.min(data)
    return list((data - np.min(data)) / _range)


def standardization(data):
    data = np.array(data)
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return list((data - mu) / sigma)


def normalization_l2(data):
    data = np.array(data)
    l2_dist = np.sqrt(sum(data**2))
    return data / l2_dist