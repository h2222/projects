#-*- coding: utf-8 -*-
# todo 1. 结果为什么是随机的
#      2. 从配对结果中去重
import numpy as np
import pymysql
from exceptions import ProcessIdNotFoundError
from conf import model_param
from preprocessing import MsgParser
from word_embedding import WordEmbedding
from DBUtils.PooledDB import PooledDB
import pandas as pd
from collections import defaultdict
from dm import DM_Module


def get_intents_index_and_input_mask(intents, corpus):
    idx = corpus['type'].isin(intents)
    # print(np.where(idx)[0].shape)
    
    input_mask = (1 - idx).astype(bool)
    return np.array(range(corpus.shape[0]))[idx], np.array(range(corpus.shape[0]))[input_mask]


class Corpus(object):
    def __init__(self, sentence, weights, intent):
        self.sentence = sentence
        self.weights = weights
        self.intent = intent


class CorpusID(object):
    def __init__(self, intents, index, input_mask):
        self.intents = intents
        self.index = index
        self.input_mask = input_mask


class CorpusDB(object):
    """SemanticTree class."""
    host = model_param['fileParam']['database']['host']
    user = model_param['fileParam']['database']['user']
    passwd = model_param['fileParam']['database']['password']
    db_name = model_param['fileParam']['database']['database']
    corpus_table = model_param['fileParam']['database']['corpus_table']
    proc_table = model_param['fileParam']['database']['proc_table']
    tokenizer = MsgParser()
    embedding = WordEmbedding()

    def __init__(self):
        self.index2corpus = []  # 保存句向量_id对应的语料
        self.intent2id = defaultdict(list)  # 保存不同意图对应的id
        corpus_matrix = []  # 保存句向量
        self.dm = DM_Module()
        # 建立数据库连接池
        pool = PooledDB(pymysql, 5, host=self.host, user=self.user, passwd=self.passwd,
                        db=self.db_name, port=3306, charset="utf8")  # 5为连接池里的最少连接数
        conn = pool.connection()

        # 读取语料库
        #corpus_df = pd.read_sql("select * from {} where mask <> 1".format(self.corpus_table), con=conn)
        corpus_df = pd.read_csv('corpus.csv')
        #corpus_df.to_csv('corpus.csv', index=0)
        print('finally')
        corpus_df['id'] = list(range(corpus_df.shape[0]))
        corpus_df = corpus_df.drop_duplicates(subset=['type', 'corpus']).reset_index()
        # 读取流程列表
        #procs = pd.read_sql("select * from {}".format(self.proc_table), con=conn)
        procs = pd.read_csv('procs.csv')
        procs['in_node'] = procs['in_node'].astype(np.str)
        #procs.to_csv('procs.csv', index=0)
        conn.close()

        # 读取流程管理词典，用于获取当前 pid - innode 下可用的 意图与对应的语料id
        self.dm_dct = {}
        proc_group = procs.groupby('processid')
        for pid, grp in proc_group:
            self.dm_dct[pid] = {}
            for in_node, gp in grp.groupby('in_node'):
                self.dm_dct[pid][in_node] = []
                intents = []
                for i, row in gp.iterrows():
                    intents.extend(row.semantic_type.split('+'))
                # print(pid, in_node, intents)
                idx, input_mask = get_intents_index_and_input_mask(intents, corpus_df)
                self.dm_dct[pid][in_node] = CorpusID(intents=intents,
                                                     index=idx,
                                                     input_mask=input_mask)
                    # for st in row.semantic_type.split('+'):
                    #     self.dm_dct[pid][in_node].append(st)
        
        # 处理语料，填充 self.index2corpus、self.intent2id、self.corpus_matrix
        for i, row in corpus_df.iterrows():
            intent = row.type
            sentence = row.corpus
            # 分词、去停用词
            tokens = self.tokenizer.tokenize(sentence, False, 'model')
            tokens_ = []
            # 按分词结果得到每个token的embedding
            for t in tokens:
                # 如果有token对应的embedding，则添加进tokens_
                if t in self.embedding:
                    tokens_.append(t)
                # 如果没有，则尝试截取token获取embedding
                else:
                    t_cut = []
                    for j in range(len(t), 0, -1):
                        if t[: j] in self.embedding:
                            t_cut.append(t[: j])
                            if t[j:] in self.embedding:
                                t_cut.append(t[j:])
                            break
                    tokens_.extend(t_cut)
            # 去除tokenss_中重复的token，比如“好的好的好的”这种，只留下“好的”
            tokens = self.tokenizer.normalize(tokens_)
            # 如果token为空，初始化一个0向量作为token的embedding
            if not tokens:
                sentence_vector = np.zeros(self.embedding.vector_size) + 0.00001
                weights = []
                self.index2corpus.append(Corpus(sentence, weights, intent))
                self.intent2id[intent].append(i)
                corpus_matrix.append(sentence_vector)
            # 如果token不为空
            else:
                # 带权重的sentence embedding
                sentence_vector = self.embedding.get_weighted_sentence_vector(tokens)
                # 计算token的cnt*tf*idf
                weights = self.embedding.get_weight(tokens, attention=True, return_words=True)
                #weights = None
                self.index2corpus.append(Corpus(sentence, weights, intent))
                self.intent2id[intent].append(i)
                corpus_matrix.append(sentence_vector)  # 添加句向量
        self.corpus_size = len(corpus_matrix)
        self.corpus_matrix = np.array(corpus_matrix)

    def match(self, pid, innode, sentence):
        """Return most similar Corpus."""

        # 对句子进行jieba切词
        tokens = self.tokenizer.tokenize(sentence, False, 'model')

        # 对切词的结果进行处理，如果词不在embedding中，进行二次切分
        tokens_ = []
        for t in tokens:
            if t in self.embedding:
                tokens_.append(t)
            else:
                t_cut = []
                for i in range(len(t), 0, -1):
                    if t[: i] in self.embedding:
                        t_cut.append(t[: i])
                        if t[i:] in self.embedding:
                            t_cut.append(t[i:])
                        break
                tokens_.extend(t_cut)
        tokens = self.tokenizer.normalize(tokens_)
        # 空向量
        if not tokens:
            vector = np.random.uniform(-0.000001, 0.000001, self.embedding.vector_size)
        else:
            # 得到句向量
            vector = self.embedding.get_weighted_sentence_vector(tokens)
        # 计算相似度
        try:
            candidates_intent = self.dm_dct[pid][innode]
        except KeyError:
            raise ProcessIdNotFoundError(pid)
        # TODO: candidates_index 和 input_mask 加入初始化的过程中
        # 有哪些语料对应的index需要保留
        candidates_index = candidates_intent.index
        # print(candidates_index)
        # input_mask = candidates_intent.input_mask
        # for intent in candidates_intent:
        #     candidates_index.extend(self.intent2id[intent])
        #
        # 如果语料的相似度不在对应的intent里面，对它进行修正
        # input_mask = np.array([1 if i in candidates_index else 0 for i in range(self.corpus_size)])

        # 基于word2vec计算余弦相似度
        similarities_k = self.embedding.cosineSimilarities(vector, self.corpus_matrix[candidates_index])
        similarities_index = np.array(list(zip(candidates_index, similarities_k)))
        # 取相似度大于0.75的
        index_k = similarities_index[:, 1] >= 0.75
        id2corpus_k = similarities_index[index_k][:, 0].astype(int)
        sim = similarities_index[index_k][:,1]
        corpus = []
        #如果大于0.75的不存在返回false        
        #if not len(id2corpus_k):
        #    return '', '', '', False
        if len(id2corpus_k)<10:
            sim  = []
            index_10 = np.argsort(similarities_index[:, 1])
            sentences = []
            sentence_index = []
            #corpus.extend([self.index2corpus[int(ind)] for ind in similarities_index[index_10[-10:], 0]])
            #sim.extend(similarities_index[index_10[-10:], 1])
            for i in index_10[-10:]:
                corpus.append(self.index2corpus[int(similarities_index[i][0])])
                sim.append(similarities_index[i][1])
            #return corpus, bert_tokens, sim
        else:
            corpus = [self.index2corpus[index] for index in id2corpus_k]
            #return corpus, bert_tokens, sim
       
        intents = [i.intent for i in corpus]
        outnodes = [self.dm.predict(pid, innode, i) for i in intents]
        corpuses = [i.sentence for i in corpus]
        # print('out_nodes', outnodes)
        # print('intens', intents)
        # print('corpus', corpuses)
        return intents, outnodes, corpuses

    def to_json(self, corpus, tokens, similarity):
        return {'corpus': corpus.sentence,
                'token': tokens,
                'sim_token': corpus.weights,
                'intent': corpus.intent,
                'similarity': similarity}


if __name__ == '__main__':
    db_param = {
        'host': "192.168.162.192",
        'user': "robot_collection",
        'password': "robot_collection_20180523",
        'database': "robot",
        'corpus_table': "corpus_test"
    }
    a = CorpusDB()
    b = a.match('benrenshoucui', '1.1', '你搞错了')
    print(b)
