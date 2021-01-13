# -*- coding: utf-8 -*-
import sys
import time
from omall.dataset.init_spark import init_spark
import json
import os


def get_daily_stat_data(dt):
    data = sc.textFile("oss://opay-datalake/algo_migration/omall/click_model_v2/dt=%s" % dt) \
        .map(json.loads).filter(lambda x: x is not None) \
        .map(lambda x: (int(x["user_id"]), (int(x["good_id"]), int(x["label"])))) \
        .groupByKey().mapValues(list).collect()
    data_dict = {}
    for elem in data:
        user_id, good_list = elem
        show_good = [x[0] for x in good_list if x[1] == 0]
        click_good = [x[0] for x in good_list if x[1] == 1]
        if user_id not in data_dict:
            data_dict[user_id] = {}
        data_dict[user_id]["show_goods_ids"] = show_good
        data_dict[user_id]["click_goods_ids"] = click_good
    return data_dict


def get_rec_data(dt):
    rec_dict = {}
    f_path = "/data/omall_rec_%s.txt" % dt
    if not os.path.exists(f_path):
        return rec_dict
    f = open(f_path)
    for line in f:
        tmp = json.loads(line)
        if tmp["type"] == "u2g" and "user_id" in tmp and "algo" in tmp:
            uid = int(tmp["user_id"])
            algo = tmp["algo"]
            if uid not in rec_dict:
                rec_dict[uid] = {}
            rec_dict[uid][algo] = tmp["goods_id"]
    return rec_dict


def get_item_rec_data(dt):
    rec_dict = {}
    f_path = "/data/omall_rec_%s.txt" % dt
    if not os.path.exists(f_path):
        return rec_dict
    f = open(f_path)
    for line in f:
        tmp = json.loads(line)
        if tmp["type"] == "g2g" and "item_id" in tmp and "algo" in tmp:
            gid = int(tmp["item_id"])
            algo = tmp["algo"]
            if gid not in rec_dict:
                rec_dict[gid] = {}
            rec_dict[gid][algo] = tmp["goods_id"]
    return rec_dict


def percent_format(x):
    return "{:.2%}".format(x)


def get_valid_ids_info(dt):
    raw_valid_ids = json.loads(open("/data/omall_b2c_nlp/b2c_valid_goods_%s.txt" % dt).readline())
    valid_ids = {}
    for elem in raw_valid_ids.items():
        gid, gc = elem
        valid_ids[int(gid)] = gc
    return valid_ids


def recall_algo_same_cate_stat(recall_dt):
    rec_data = get_item_rec_data(recall_dt)
    valid_ids_data = get_valid_ids_info(recall_dt)
    all_stat = {}
    for gid in rec_data:
        for algo in rec_data[gid]:
            if algo not in all_stat:
                all_stat[algo] = [0, 0]
            gc = valid_ids_data.get(gid, 0)
            if gc > 0:
                gc_same_ids = [x for x in rec_data[gid][algo] if valid_ids_data.get(x, 0) == gc]
                rec_len = len(rec_data[gid][algo])
                gc_same_ids_len = len(gc_same_ids)
                gc_rate = gc_same_ids_len / float(rec_len) if rec_len > 0 else 0
                print "algo:%s, good_id: %d, rec_len: %d, same_category_len: %d, same_category_rate: %s" % \
                      (algo, gid, rec_len, gc_same_ids_len, percent_format(gc_rate))
                all_stat[algo][0] += rec_len
                all_stat[algo][1] += gc_same_ids_len
    for algo in all_stat:
        algo_rate = all_stat[algo][1] / float(all_stat[algo][0]) if all_stat[algo][0] > 0 else 0
        print "algo: %s, all_rec_num: %d,  same_category_num:%d, same_rate: %s" % \
              (algo, all_stat[algo][0], all_stat[algo][1], percent_format(algo_rate))


def recall_algo_stat(recall_dt):
    stat_data = get_daily_stat_data(recall_dt)
    rec_data = get_rec_data(recall_dt)
    all_stat = {}
    for uid in rec_data:
        if uid in stat_data:
            show_ids = set(stat_data[uid]["show_goods_ids"])
            click_ids = set(stat_data[uid]["click_goods_ids"])
            show_len = len(show_ids)
            click_len = len(click_ids)
            for algo in rec_data[uid]:
                algo_ids = rec_data[uid][algo]
                show_num = len(set(algo_ids) & show_ids)
                click_num = len(set(algo_ids) & click_ids)
                rank_pass_rate = show_num / float(show_len) if show_len > 0 else 0
                click_rate = click_num / float(click_len) if click_len > 0 else 0
                print "user_id: %d, algo: %s,  rank_pass_rate:%s,  click_rate:%s" % \
                      (uid, algo, percent_format(rank_pass_rate), percent_format(click_rate))
                if algo not in all_stat:
                    all_stat[algo] = [0, 0, 0, 0]
                all_stat[algo][0] += show_num
                all_stat[algo][1] += click_num
                all_stat[algo][2] += show_len
                all_stat[algo][3] += click_len
    for algo in all_stat:
        if len(all_stat[algo]) == 4:
            rank_pass_rate = all_stat[algo][0] / float(all_stat[algo][2]) if all_stat[algo][2] > 0 else 0
            click_rate = all_stat[algo][1] / float(all_stat[algo][3]) if all_stat[algo][3] > 0 else 0
            print "algo: %s, rank_pass_rate:%s,  click_rate:%s" % \
                  (algo, percent_format(rank_pass_rate), percent_format(click_rate))


if __name__ == "__main__":
    spark, sc = init_spark('model_perform_check')
    sc.setLogLevel("WARN")
    check_dt = str(sys.argv[1]) if len(sys.argv) > 1 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    recall_algo_same_cate_stat(check_dt)
