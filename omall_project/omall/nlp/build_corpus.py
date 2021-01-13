# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import re
import enchant
import time
import os
from utils import *
import itertools
from textblob import Word
from tqdm import tqdm
import sys
import redis
reload(sys)
sys.setdefaultencoding('utf8')

d = enchant.Dict("en_US")


def str_min_distance(word1, word2):
    global edit_distance_cache
    import sys
    reload(sys)
    sys.setdefaultencoding('utf8')
    peer = [word1, word2]
    peer.sort()
    peer_key = "|".join(peer)
    if peer_key in edit_distance_cache:
        return edit_distance_cache[peer_key]
    n = len(word1)
    m = len(word2)
    max_score = 100
    if abs(n - m) >= 5:
        return max_score
    # 有一个字符串为空串
    if n * m == 0:
        if peer_key not in edit_distance_cache:
            edit_distance_cache[peer_key] = n + m
        return n + m
    if word1 == word2:
        return 0
    if len(set([x for x in word1]) & set([x for x in word2])) == 0:
        if peer_key not in edit_distance_cache:
            edit_distance_cache[peer_key] = max(n, m)
        return max(n, m)
    # DP 数组
    D = [[0] * (m + 1) for _ in range(n + 1)]
    # 边界状态初始化
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    # 计算所有 DP 值
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = D[i - 1][j] + 1
            down = D[i][j - 1] + 1
            left_down = D[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            D[i][j] = min(left, down, left_down)
    prex_score = 0
    for i in range(min(n, m)):
        if word1[i] == word2[i]:
            prex_score -= 0.1
        else:
            break
    D[n][m] -= prex_score
    if peer_key not in edit_distance_cache:
        edit_distance_cache[peer_key] = D[n][m]
    return D[n][m]


# brand_name_path = "./brand.txt"
brand_name_path = "/home/hdfs/algo-offline-job/omall/nlp/brand.txt"


def get_brand_names():
    arr = []
    if not os.path.exists(brand_name_path):
        return arr
    f = open(brand_name_path)
    for line in f:
        arr.append(line.lower().strip())
    return arr


goods_name_query = """
select id, update_time, brand_id, category_id, name, keywords, goods_brief, goods_desc from otrade_dw.dwd_otrade_b2c_mall_nideshop_goods_hf where dt = '{dt}'
"""

category_query = """
select id, name from otrade_dw.dwd_otrade_b2c_mall_nideshop_category_hf where dt = "{dt}"
"""

brand_query = """
select id, name from otrade_dw_ods.ods_binlog_mall_nideshop_brand_all_hi where id is not null
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


def no_html(x):
    try:
        raw_str = re.sub('<[^<]+?>', '', x).replace('\n', '').strip().lower()
        return raw_str
    except:
        pass


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
    return filter_special_str(split_digit(no_html(x)))


def mapper(x):
    try:
        return int(x.id), str2arr(pure_str(none_str(x.name)))
    except:
        pass


def str2arr(x):
    res = []
    if x == "":
        return res
    arr = [i for i in x.split(" ") if len(i) >= 1]
    for elem in arr:
        if elem not in res:
            res.append(elem)
    return res


def goods_mapper(x):
    try:
        return int(x.id), (x.update_time, int(x.brand_id), int(x.category_id),
                           str2arr(pure_str(none_str(x.name))), str2arr(pure_str(none_str(x.keywords))),
                           str2arr(pure_str(none_str(x.goods_brief))), str2arr(pure_str(none_str(x.goods_desc))))
    except:
        pass


def nlp_stat(candidate_words, w2s):
    f = get_fp("/data/omall_b2c_nlp/b2c_nlp_error_report_%s.txt" % dt)
    error_word_num = 20
    stat_dict = {}
    for w in tqdm(candidate_words):
        has_group = False
        stat_key = stat_dict.keys()
        for sk in stat_key:
            if str_min_distance(w, sk) <= 2:
                has_group = True
                break
        if has_group:
            continue
        cur_scores = []
        for ow in candidate_words:
            cur_scores.append(str_min_distance(w, ow))
        tmp_data = zip(candidate_words, cur_scores)
        tmp_data.sort(key=lambda x: x[-1])
        res_words = [x[0] for x in tmp_data if 0 < x[1] <= 4]
        if len(res_words) > 0:
            stat_dict[w] = res_words[:min(error_word_num, len(res_words))]
            f.write(json.dumps({
                "origin_word": w,
                "neighbor_word": res_words[:min(error_word_num, len(res_words))],
                "source": [w2s.get(e, "") for e in res_words[:min(error_word_num, len(res_words))]]
            }))
            f.write("\n")
    print len(stat_dict)
    f.close()
    send_email(dt + 'b2c_omall_spell_error', receivers, "", ["/data/omall_b2c_nlp/b2c_nlp_error_report_%s.txt" % dt])


def get_fp(f_path):
    if os.path.exists(f_path):
        os.system("rm -rf %s" % f_path)
    return open(f_path, "aw")


def write_goodname_res2rds(raw_data):
    rds_host = "r-d7oven45vpwhoe42ev.redis.eu-west-1.rds.aliyuncs.com"
    rds_port = 6379
    rds = redis.StrictRedis(host=rds_host, port=rds_port)
    rpipe = rds.pipeline(transaction=False)
    expire_time = 86400 * 7
    #
    pipe_command_num = 64
    counter = 0
    key_format = "om_goods_name_ids:%s"
    for sw in raw_data:
        try:
            key = key_format % sw
            print sw, key, len(raw_data[sw])
            rpipe.set(key, json.dumps({
                "good_ids": raw_data[sw],
            }), ex=expire_time)
            counter += 1
            if counter > 0 and counter % pipe_command_num == 0:
                time.sleep(0.5)
                rpipe.execute()
        except Exception as e:
            print e.message
            pass
    rpipe.execute()
    rpipe.close()


def words2goods(gdata):
    """
    reverse sort between good_id 2 words
    assume the search words of users are 2 or 3
    :return:
    """
    goods_name_res = {}
    yt = time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    valid_ids = set(map(int, json.loads(open("/data/omall_b2c_nlp/b2c_valid_goods_%s.txt" % yt).readline()).keys()))
    f = get_fp("/data/omall_b2c_nlp/b2c_goods_name_recall_%s.txt" % dt)
    res_dict = {}
    words_num = [1, 2, 3, 4]
    for gid in gdata:
        gwords = list(set(gdata[gid]["name_words"] + gdata[gid]["brand_words"] + gdata[gid]["category_words"]))
        gwords.sort()
        for wn in words_num:
            if wn not in res_dict:
                res_dict[wn] = {}
            it = itertools.combinations(gwords, wn)
            for i in it:
                t = " ".join(list(i))
                if t not in res_dict[wn]:
                    res_dict[wn][t] = []
                if gid not in res_dict[wn][t]:
                    res_dict[wn][t].append(gid)
    res_data = []
    for wn in res_dict:
        res_data += [x for x in zip([wn] * len(res_dict[wn]), res_dict[wn].keys(), res_dict[wn].values()) if len(x[-1]) > 2]
    print len(res_data)
    res_data.sort(key=lambda x: (x[0], len(x[-1])), reverse=True)
    for elem in res_data:
        f.write(json.dumps({
            "search_words": elem[1],
            "good_ids": elem[2]
        }))
        goods_ids = list(set(elem[2]) & set(valid_ids))
        if len(goods_ids) > 0 and elem[1] not in goods_name_res:
            goods_name_res[elem[1]] = goods_ids[:min(30, len(goods_ids))]
        f.write("\n")
    f.close()
    print len(goods_name_res)
    write_goodname_res2rds(goods_name_res)


def singularize_word(word):
    global singularize_cache
    try:
        if word in singularize_cache:
            return singularize_cache[word]
        res = str(Word(word).singularize())
        if word not in singularize_cache:
            singularize_cache[word] = res
    except Exception as e:
        print e.message
        return word
    return res


def get_data(nlp_dt):
    print nlp_dt
    brand_arr = get_brand_names()
    goods_name_data = spark.sql(goods_name_query.format(dt=nlp_dt)).rdd.map(goods_mapper) \
        .filter(lambda x: x is not None).reduceByKey(max)
    category_data = spark.sql(category_query.format(dt=nlp_dt)).rdd.map(mapper).filter(lambda x: x is not None)
    brand_data = spark.sql(brand_query).rdd.map(mapper).filter(lambda x: x is not None)
    ##
    goods_name_arr = goods_name_data.collect()
    category_dict = dict(category_data.collect())
    brand_dict = dict(brand_data.collect())
    print len(goods_name_arr), len(category_dict), len(brand_dict)
    all_right_words = set(brand_arr)
    words2source = {}
    for cid in category_dict:
        category_dict[cid] = map(singularize_word, category_dict[cid])
        for elem in category_dict[cid]:
            if elem not in words2source:
                words2source[elem] = "category:{cid}".format(cid=cid)
        all_right_words.update(category_dict[cid])
    for bid in brand_dict:
        all_right_words.update(brand_dict[bid])
        for elem in brand_dict[bid]:
            if elem not in words2source:
                words2source[elem] = "brand:{bid}".format(bid=bid)
    f_path = "/data/omall_b2c_nlp/goods_nlp_%s.txt" % dt
    f = get_fp(f_path)
    goods_dict = {}
    for elem in goods_name_arr:
        (gid, (_, bid, cid, name, keywords, goods_brief, goods_desc)) = elem
        if gid not in goods_dict:
            name = [w for w in map(singularize_word, name) if w not in meaningless_words]
            keywords = [w for w in map(singularize_word, keywords) if w not in meaningless_words]
            goods_brief = [w for w in map(singularize_word, goods_brief) if w not in meaningless_words]
            goods_desc = [w for w in map(singularize_word, goods_desc) if w not in meaningless_words]
            brand_words = brand_dict.get(bid, [])
            category_words = category_dict.get(cid, [])
            goods_dict[gid] = {
                "name_words": name,
                "keyword_words": keywords,
                "brief_words": goods_brief,
                "desc_words": goods_desc,
                "brand_words": brand_words,
                "category_words": category_words,
            }
            all_right_words.update(name + keywords + goods_brief + goods_desc)
            for e in name + keywords + goods_brief + goods_desc:
                if e not in words2source:
                    words2source[e] = "good_id:{gid}".format(gid=gid)
    f.write(json.dumps(goods_dict))
    f.close()
    new_all_right_words = []
    for rw in all_right_words:
        try:
            if d.check(rw) is False:
                new_all_right_words.append(rw)
        except Exception as e:
            print e.message
            new_all_right_words.append(rw)
            pass
    new_all_right_words.sort()
    print len(new_all_right_words)
    ##
    f2 = get_fp("/data/omall_b2c_nlp/b2c_nlp_right_words_%s" % nlp_dt)
    f2.write(json.dumps({
        "b2c_right_words": new_all_right_words
    }))
    f2.close()
    nlp_stat(new_all_right_words, words2source)
    ##
    words2goods(goods_dict)
    upload_via_oss(f_path, "omall_b2c_nlp/goods_words.txt")


def get_edit_distance_cache():
    ed_cache = {}
    if os.path.exists("/data/omall_b2c_nlp/min_edit_distance.txt"):
        ed_cache = json.loads(open("/data/omall_b2c_nlp/min_edit_distance.txt").readline())
    return ed_cache


def get_singularize_cache():
    sg_cache = {}
    if os.path.exists("/data/omall_b2c_nlp/b2c_singularize.txt"):
        sg_cache = json.loads(open("/data/omall_b2c_nlp/b2c_singularize.txt").readline())
    return sg_cache


if __name__ == "__main__":
    edit_distance_cache = get_edit_distance_cache()
    singularize_cache = get_singularize_cache()
    spark, sc = init_spark("build_b2c_corpus")
    sc.setLogLevel("ERROR")
    dt = sys.argv[1] if len(sys.argv) > 1 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    get_data(dt)
    f = get_fp("/data/omall_b2c_nlp/min_edit_distance.txt")
    f.write(json.dumps(edit_distance_cache))
    f.close()
    f = get_fp("/data/omall_b2c_nlp/b2c_singularize.txt")
    f.write(json.dumps(singularize_cache))
    f.close()
