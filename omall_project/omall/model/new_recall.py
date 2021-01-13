# -*- coding: utf-8 -*-
import sys
from util import *
from gensim.models.ldamodel import LdaModel
from collections import OrderedDict
from surprise import SVD
from surprise import Reader
from surprise import Dataset
import pandas as pd
from surprise.model_selection import GridSearchCV
import numpy as np

embed_size = 4


def doc2topics(raw_list):
    doc_topics = [0] * embed_size
    for elem in raw_list:
        index = elem[0]
        value = elem[1]
        doc_topics[index] = value
    return np.array(doc_topics)


def get_data(filePath):
    data_file = open(filePath)
    user_item_stat = {}
    ui_dict = OrderedDict()
    for line in data_file:
        try:
            rec = json.loads(line)
            uid = int(rec['user_id'])
            iid = int(rec['good_id'])
            cn = int(rec['click_num'])
            sn = int(rec['show_num'])
            if uid not in user_item_stat:
                user_item_stat[uid] = {}
            if iid not in user_item_stat[uid]:
                user_item_stat[uid][iid] = {}
                user_item_stat[uid][iid]["show_num"] = 0
                user_item_stat[uid][iid]["click_num"] = 0
            user_item_stat[uid][iid]["show_num"] += sn
            user_item_stat[uid][iid]["click_num"] += cn
        except:
            pass
    max_rate, min_rate = -1000000, 1000000
    for uid in user_item_stat:
        val = 0
        for iid in user_item_stat[uid]:
            val += user_item_stat[uid][iid]["click_num"]
        if val >= min_user_active:
            if uid not in ui_dict:
                ui_dict[uid] = {}
            for iid in user_item_stat[uid]:
                if iid not in ui_dict[uid]:
                    ui_dict[uid][iid] = 0
                ui_dict[uid][iid] += round(
                    user_item_stat[uid][iid]["click_num"] - user_item_stat[uid][iid]["show_num"] * 0.1, 2)
                r = ui_dict[uid][iid]
                if r > max_rate:
                    max_rate = r
                if r < min_rate:
                    min_rate = r
    _range = float(max_rate - min_rate)
    print max_rate, min_rate, _range
    if _range == 0:
        return
    data_file.close()
    uid2index = OrderedDict()
    iid2index = OrderedDict()
    for k, v in ui_dict.items():
        uid2index[k] = len(uid2index)
        for elem in v.keys():
            if elem not in iid2index:
                iid2index[elem] = len(iid2index)
    corpus = []
    for k, v in ui_dict.items():
        corpus.append([(iid2index[elem], (v[elem] - min_rate) / _range) for elem in v.keys() if len(v) >= 2])
    return corpus, uid2index, iid2index, ui_dict


def get_data_with_nlp(g2n_dict):
    word_item_dict = OrderedDict()
    for iid in g2n_dict:
        words = g2n_dict[iid]
        for w in words:
            try:
                w.decode('ascii')
                if w not in word_item_dict:
                    word_item_dict[w] = {}
                if iid not in word_item_dict[w]:
                    word_item_dict[w][iid] = 1
                else:
                    word_item_dict[w][iid] += 1
            except:
                pass
    word2index = OrderedDict()
    iid2index = OrderedDict()
    for k, v in word_item_dict.items():
        word2index[k] = len(word2index)
        for elem in v.keys():
            if elem not in iid2index:
                iid2index[elem] = len(iid2index)
    corpus = []
    for k, v in word_item_dict.items():
        corpus.append([(iid2index[elem], v[elem]) for elem in v.keys() if len(v) >= 2])
    return corpus, word2index, iid2index, word_item_dict


def get_lda_res(c, uidx, iidx, algo="lda", cal_user=True):
    print 'Users: %s' % len(c)
    print 'Items: %s' % len(iidx)
    print '********* Start LDA training *********'
    model = None
    for i in [4, 5, 6, 7, 8, 10, 16]:
        cur_model = LdaModel(c, num_topics=i, alpha='auto', eval_every=5, passes=10)
        val = cur_model.log_perplexity(c)
        print i, val
        if not np.isnan(val):
            model = cur_model
            break
    if model is None:
        return
    I = model.get_topics().transpose()
    # rec goods via similar goods
    if cal_user:
        U = np.zeros(shape=(len(c), embed_size))
        for i, elem in enumerate(c):
            U[i] = doc2topics(model.get_document_topics(elem))
        U = U.astype(np.float32)
        rec_res = get_personas_goods(U, I, uidx, iidx, valid_ids=valid_ids, max_goods_num=200)
        get_default_goods(algo, uid2goods, U, I, iidx, valid_ids=valid_ids, max_goods_num=200)
        write_algo2rds(rec_res, algo, uid2goods)
        write2localfile(f_path, algo, rec_res, uid2goods)
    item_rec_res = get_personas_goods(I, I, iidx, iidx, valid_ids=valid_ids, max_goods_num=100)
    write_algo2rds(item_rec_res, algo, goods2goods)
    write2localfile(f_path, algo, item_rec_res, goods2goods)


# 此算法perfect
def get_svd_res(ui_dict, algo="svd", cal_user=True):
    users, items, rates = [], [], []
    for uid in ui_dict:
        for item in ui_dict[uid]:
            users.append(uid)
            items.append(item)
            r = ui_dict[uid][item]
            rates.append(r)
    rates = normalization(rates)
    print min(rates), max(rates)
    ratings_dict = {'itemID': items, 'userID': users, 'rating': rates}
    del users, items
    df = pd.DataFrame(ratings_dict)
    del ratings_dict
    reader = Reader(rating_scale=(min(rates), max(rates)))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    del df
    #
    param_grid = {'n_factors': [4, 8],
                  'n_epochs': [50], 'lr_all': [0.01],
                  'reg_all': [0.1]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])
    print(gs.best_score['mae'])
    print(gs.best_params['mae'])
    model = gs.best_estimator['rmse']
    trainset = data.build_full_trainset()
    model.fit(trainset)
    #
    del data
    print 'Users: %s' % trainset.n_users
    print 'Items: %s' % trainset.n_items
    print '********* Start SVD training *********'
    U, I = model.pu, model.qi
    del model
    U = U.astype(np.float32)
    I = I.astype(np.float32)
    u2idx = trainset._raw2inner_id_users
    i2idx = trainset._raw2inner_id_items
    if cal_user:
        rec_res = get_personas_goods(U, I, u2idx, i2idx, valid_ids=valid_ids, max_goods_num=150)
        get_default_goods(algo, uid2goods, U, I, i2idx, valid_ids=valid_ids, max_goods_num=150)
        write_algo2rds(rec_res, algo, uid2goods)
        write2localfile(f_path, algo, rec_res, uid2goods)
    # rec goods via similar goods
    item_rec_res = get_personas_goods(I, I, i2idx, i2idx, valid_ids=valid_ids, max_goods_num=100)
    write_algo2rds(item_rec_res, algo, goods2goods)
    write2localfile(f_path, algo, item_rec_res, goods2goods)


if __name__ == '__main__':
    model_type = str(sys.argv[1]) if len(sys.argv) > 1 else "opg_pair_dataset"
    model_dir = '/data/omall/model_%s/' % model_type
    if os.path.exists(model_dir):
        os.system("rm -rf %s" % model_dir)
    dt = sys.argv[1] if len(sys.argv) > 1 \
        else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    valid_ids = set(map(int, json.loads(open("/data/omall_b2c_nlp/b2c_valid_goods_%s.txt" % dt).readline()).keys()))
    download_data(dt, model_type)
    path1 = "/data/omall/%s/dt=%s/part-00000" % (model_type, dt)
    corpus, uid2index, iid2index, user_item_dict = get_data(path1)
    f_name = "/data/omall_rec_%s.txt" % dt
    f_path = open(f_name, "aw")
    get_lda_res(corpus, uid2index, iid2index)
    get_svd_res(user_item_dict)
    # #覆盖的商品面过少
    nlp_dict = get_nlp_data(dt)
    if len(nlp_dict) > 0:
        nlp_corpus, nlp_uid2index, nlp_iid2index, nlp_user_item_dict = get_data_with_nlp(nlp_dict)
        get_lda_res(nlp_corpus, nlp_uid2index, nlp_iid2index, algo="nlp_lda", cal_user=False)
        get_svd_res(nlp_user_item_dict, algo="nlp_svd", cal_user=False)
    f_path.close()
