# -*- coding: utf-8 -*-
# user_id(as word) 2 good_id
# good_id desc word 2 good_id
import os
from util import *
import json
from gensim.models import Word2Vec
from collections import OrderedDict
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import redis
import sys
import json

embed_size = 4


def get_data(filePath):
    data_file = open(filePath)
    user_item_dict = OrderedDict()
    for line in data_file:
        try:
            rec = json.loads(line)
            uid = int(rec['user_id'])
            iid = int(rec['good_id'])
            cn = int(rec['click_num'])
            if cn > 0:
                if uid not in user_item_dict:
                    user_item_dict[uid] = []
                user_item_dict[uid].append(str(iid))
        except:
            pass
    data_file.close()
    corpus = [v for _, v in user_item_dict.items() if len(v) >= 2]
    iids = set()
    for sentence in corpus:
        for word in sentence:
            iids.add(word)
    return corpus, iids


def get_nlp_dataset(g2n_dict):
    word_item_dict = OrderedDict()
    for iid in g2n_dict:
        words = g2n_dict[iid]
        for w in words:
            try:
                w.decode('ascii')
                if w not in word_item_dict:
                    word_item_dict[w] = []
                word_item_dict[w].append(str(iid))
            except:
                pass
    corpus = [v for _, v in word_item_dict.items()]
    return corpus


def word2vec_recall(c, algo_name="word2vec"):
    # workers:  to speed up training
    # size: is the number of dimensions (N) of the N-dimensional space that gensim Word2Vec maps the words onto.
    print '********* Start word2vec training *********'
    model = Word2Vec(c, size=embed_size, window=5, min_count=1, workers=4, sg=0, hs=0, compute_loss=True)
    training_loss = model.get_latest_training_loss()
    print training_loss
    item_rec_res = {}
    word_vectors = model.wv
    for i, word in enumerate(word_vectors.vocab):
        candidate_ids = map(int, [x[0] for x in word_vectors.most_similar(word, topn=50)])
        item_rec_res[int(word)] = [x for x in candidate_ids if x in valid_ids]
    write_algo2rds(item_rec_res, algo_name, goods2goods)
    write2localfile(f_path, algo_name, item_rec_res, goods2goods)


if __name__ == '__main__':
    model_type = str(sys.argv[1]) if len(sys.argv) > 1 else "opg_pair_dataset"
    model_dir = '/data/omall/model_%s/' % model_type
    if os.path.exists(model_dir):
        os.system("rm -rf %s" % model_dir)
    dt = sys.argv[1] if len(sys.argv) > 1 \
        else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    download_data(dt, model_type)
    path1 = "/data/omall/%s/dt=%s/part-00000" % (model_type, dt)
    valid_ids = set(map(int, json.loads(open("/data/omall_b2c_nlp/b2c_valid_goods_%s.txt" % dt).readline()).keys()))
    corpus, iids_all = get_data(path1)
    iids_all = list(iids_all)
    #
    f_name = "/data/omall_rec_%s.txt" % dt
    f_path = open(f_name, "aw")
    #
    word2vec_recall(corpus)
    nlp_dict = get_nlp_data(dt)
    if len(nlp_dict) > 0:
        nlp_corpus = get_nlp_dataset(nlp_dict)
        word2vec_recall(nlp_corpus, "nlp_word2vec")
    #
    f_path.close()
    upload_via_oss(f_name, "omall_rec/goods.txt")
    remove_data(dt, model_type)

