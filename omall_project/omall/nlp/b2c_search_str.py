# -*- coding: utf-8 -*-
from init_spark import init_spark
import json
import time
from operator import add
import enchant
import redis
import re
from textblob import Word
from utils import *
from tqdm import tqdm
from datetime import datetime, timedelta
import sys
reload(sys)
sys.setdefaultencoding('utf8')


d = enchant.Dict("en_US")


############################################


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


def pair_edit_distance(word1_arr, word2_arr):
    s_arr, l_arr = word1_arr, word2_arr
    if len(word1_arr) > len(word2_arr):
        s_arr, l_arr = word2_arr, word1_arr
    score_arr = []
    for sw in s_arr:
        score_arr.append(min([str_min_distance(sw, x) for x in l_arr]))
    limit_gap = 5
    if len(s_arr) == 1:
        if len(s_arr[0]) <= 3:
            limit_gap = 0
        else:
            limit_gap = 1
    else:
        small_word_len = sum([len(x) for x in s_arr])
        limit_gap = min([small_word_len * 0.5, limit_gap])
    if sum(score_arr) <= limit_gap:
        return sum(score_arr) / float(len(score_arr)) if len(score_arr) > 0 else 0
    return None


############################################
rds_host = "r-d7oven45vpwhoe42ev.redis.eu-west-1.rds.aliyuncs.com"
rds_port = 6379

# show & click & into
query_opay_client = """
select message from oride_source.opay_ep_logv1 where dt >= '{dt}'
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
    if x.startswith("i"):
        for elem in sc_replace_arr.value:
            x = x.replace(elem[0], elem[1])
    return filter_special_str(split_digit(no_html(x)))


def str2arr(x):
    res = []
    if x == "":
        return res
    arr = [i for i in x.split(" ") if len(i) >= 1]
    for elem in arr:
        if elem not in res:
            res.append(elem)
    return res


def opay_client_mapper(x):
    try:
        ret = []
        tmp = json.loads(x.message)
        events = tmp["es"]
        for e in events:
            event_name = e['en'].strip()
            val = e['ev']
            # category_id = val.get('category_id', '')
            keyword = pure_str(val.get('keyword', ''))
            if keyword != "":
                if event_name == "TAB_MALL_Searchresultpage_haveresults_show":
                    ret.append((("haveres", keyword), 1))
                elif event_name == "TAB_MALL_Searchresultpage_noresults_show":
                    ret.append((("nores", keyword), 1))
        return ret
    except:
        pass


def get_right_words():
    right_words = json.loads(open("/data/omall_b2c_nlp/b2c_nlp_right_words_%s" % yt).readline())["b2c_right_words"]
    return right_words


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


def rdd_stat(r):
    data = r.collect()
    arr = []
    for elem in tqdm(data):
        arr.append([w for w in map(singularize_word, str2arr(elem)) if w not in meaningless_words])
    return arr


def combineword_logic(word1, word2):
    arr1 = word1.split(" ")
    arr2 = word2.split(" ")
    a1, a2 = [], []
    for e in arr1:
        if e not in a1:
            a1.append(e)
    for e in arr2:
        if e not in a2:
            a2.append(e)
    a1.sort()
    a2.sort()
    if "".join(a1) == "".join(a2):
        return True
    return False


def write2rds(raw_data):
    rds = redis.StrictRedis(host=rds_host, port=rds_port)
    rpipe = rds.pipeline(transaction=False)
    expire_time = 86400 * 7
    #
    pipe_command_num = 64
    counter = 0
    key_format = "om_search_mdf:%s"
    for wrong_word in raw_data:
        right_word = raw_data[wrong_word]
        key = key_format % right_word
        print key, wrong_word, right_word
        rpipe.set(key, right_word, ex=expire_time)
        counter += 1
        if counter > 0 and counter % pipe_command_num == 0:
            time.sleep(0.5)
            rpipe.execute()
    rpipe.execute()
    rpipe.close()
    for elem in cover_pair:
        (wrong_word, right_word) = elem
        print key_format % right_word, wrong_word, right_word
        rds.set(key_format % wrong_word, right_word, ex=expire_time)


def nores_adjust(nores_data, haveres_data):
    right_words = get_right_words()
    print "adjust the no res input to those have res"
    haveres_keyword = set()
    for t in haveres_data:
        update_t = [x for x in t if len(x) > 1]
        haveres_keyword.update(update_t)
    print len(haveres_keyword)
    all_right_words = haveres_keyword | set(right_words)
    print len(all_right_words)
    adjust_data = {}
    nores_modify_arr = []
    for words_arr in tqdm(nores_data):
        modify_arr = words_arr[:]
        for widx in range(len(words_arr)):
            w = words_arr[widx]
            # the length difference between two words should be less than 4
            # first alpha must be same, combine word use suggestion, max edit distance should be 4
            if len(w) > 2 and w not in all_right_words and w.isalpha() and not d.check(w):
                res_word = ""
                if w in cover_dict:
                    res_word = cover_dict[w]
                else:
                    sug_words = d.suggest(w)
                    sug_words = ["".join(i.lower().split(" ")).replace("-", "") for i in sug_words if " " not in i]
                    candidate_words = [x for x in all_right_words if
                                       abs(len(w) - len(x)) <= 3 and w[0] == x[0]] + sug_words
                    candidate_scores = []
                    for cw in candidate_words:
                        bonus = str_min_distance(w, cw)
                        if cw in right_words:
                            bonus -= 1
                        candidate_scores.append(bonus)
                    tmp_data = zip(candidate_words, candidate_scores)
                    tmp_data.sort(key=lambda x: x[-1])
                    if len(tmp_data) > 0:
                        res_word = tmp_data[0][0]
                if res_word != "":
                    modify_arr[widx] = res_word
                    try:
                        print "origin words: %s,  adjust word: %s, res_word: %s" % (words_arr, w, res_word)
                        if w not in adjust_data:
                            adjust_data[w] = res_word
                    except Exception as e:
                        print e.message
                        pass
        nores_modify_arr.append(modify_arr)
    if len(adjust_data) > 0:
        print len(adjust_data)
        write2rds(adjust_data)
    print "nores_modify_arr: %d" % len(nores_modify_arr)
    return nores_modify_arr


def write_search_word2rds(raw_data):
    rds = redis.StrictRedis(host=rds_host, port=rds_port)
    rpipe = rds.pipeline(transaction=False)
    expire_time = 86400 * 7
    #
    pipe_command_num = 64
    counter = 0
    key_format = "om_search_goods_res:%s"
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


def get_search_word_data(d_range=15):
    # 热门点击召回
    data = sc.parallelize([])
    yt = time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    dt = datetime.strptime(yt, '%Y-%m-%d')
    for i in range(d_range):
        try:
            cur_dt = dt - timedelta(days=i)
            ret = sc.textFile(
                "oss://opay-datalake/algo_migration/omall/b2c_search/dt=%s/*" % cur_dt.strftime('%Y-%m-%d'))
            data = data.union(ret)
            print cur_dt
        except:
            pass
    data = data.map(json.loads).filter(lambda x: x is not None)\
        .map(lambda x: (x["st"], x["keyword"])).filter(lambda x: x is not None).distinct()
    return data


def search_res(d_range):
    data = get_search_word_data(d_range)
    haveres = data.filter(lambda (x, y): x == "y")
    haveres_keyword = rdd_stat(haveres.map(lambda (x, y): y))
    nores = data.filter(lambda (x, y): x == "n")
    nores_keyword = rdd_stat(nores.map(lambda (x, y): y))
    print "no_res_word", nores_keyword
    nores_modify = nores_adjust(nores_keyword, haveres_keyword)
    all_words = []
    for elem in haveres_keyword + nores_modify:
        if elem != [] and elem not in all_words:
            all_words.append(elem)
    print len(all_words)
    all_words.sort()
    valid_ids = set(map(int, json.loads(open("/data/omall_b2c_nlp/b2c_valid_goods_%s.txt" % yt).readline()).keys()))
    tmp_goods_name_data = json.loads(open("/data/omall_b2c_nlp/goods_nlp_%s.txt" % yt).readline())
    goods_name_data = {}
    for gid in tmp_goods_name_data:
        int_gid = int(gid)
        if int_gid in valid_ids and tmp_goods_name_data[gid]["name_words"] != []:
            goods_name_data[int_gid] = tmp_goods_name_data[gid]["name_words"]
    search_words_res_dict = {}
    f_str_res = get_fp("/data/omall_b2c_nlp/b2c_search_word_res_%s.txt" % yt)
    for search_keyword in tqdm(all_words):
        csearch_words = []
        csearch_scores = []
        csearch_gids = []
        for gid in goods_name_data:
            s = pair_edit_distance(search_keyword, goods_name_data[gid])
            if s is not None:
                csearch_words.append(goods_name_data[gid])
                csearch_scores.append(s)
                csearch_gids.append(gid)
        if len(csearch_words) > 0:
            csearch_res = zip(csearch_words, csearch_gids, csearch_scores)
            csearch_res.sort(key=lambda x: x[-1])
            csearch_res = csearch_res[:min(30, len(csearch_res))]
            cids = [x[1] for x in csearch_res]
            if len(cids) > 0:
                try:
                    search_keyword.sort()
                    str_search_word = " ".join(search_keyword)
                    search_words_res_dict[str_search_word] = cids
                    f_str_res.write(json.dumps({
                        "keyword": str_search_word,
                        "goods_names": [x[0] for x in csearch_res]
                    }))
                    f_str_res.write("\n")
                except Exception as e:
                    print e.message
                    pass
    write_search_word2rds(search_words_res_dict)
    f_str_res.close()


def get_edit_distance_cache():
    ed_cache = {}
    if os.path.exists("/data/omall_b2c_nlp/min_edit_distance.txt"):
        ed_cache = json.loads(open("/data/omall_b2c_nlp/min_edit_distance.txt").readline())
    return ed_cache


def get_singularize_cache():
    s_cache = {}
    if os.path.exists("/data/omall_b2c_nlp/b2c_singularize.txt"):
        s_cache = json.loads(open("/data/omall_b2c_nlp/b2c_singularize.txt").readline())
    return s_cache


if __name__ == "__main__":
    edit_distance_cache = get_edit_distance_cache()
    singularize_cache = get_singularize_cache()
    print len(edit_distance_cache), len(singularize_cache)
    spark, sc = init_spark("b2c_search")
    sc.setLogLevel("ERROR")
    sc_replace_arr = sc.broadcast(replace_arr)
    yt = time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    day_range = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    print yt
    search_res(day_range)
    #
    f = get_fp("/data/omall_b2c_nlp/min_edit_distance.txt")
    f.write(json.dumps(edit_distance_cache))
    f.close()
    #
    f = get_fp("/data/omall_b2c_nlp/b2c_singularize.txt")
    f.write(json.dumps(singularize_cache))
    f.close()
