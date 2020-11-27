# -*- coding: utf-8 -*-
import numpy as np
from exceptions import ProcessIdNotFoundError
import pandas as pd
from collections import defaultdict
from dm import DM_Module
from bert_serving.client import BertClient
import jieba


np.random.seed(2019)


def get_intents_index_and_input_mask(intents, corpus):
    idx = corpus["type"].isin(intents)
    input_mask = (1 - idx).astype(bool)
    return (
        np.array(range(corpus.shape[0]))[idx],
        np.array(range(corpus.shape[0]))[input_mask],
    )


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


class CorpusBert(object):
    """SemanticTree class."""

    def __init__(self, corpus_path, tokenizer=False):
        self.index2corpus = []  # 保存句向量_id对应的语料
        self.intent2id = defaultdict(list)  # 保存不同意图对应的id
        self.corpus_matrix = []  # 保存句向量
        self.dm = DM_Module()
        self.tokenizer = tokenizer
        if tokenizer:
            dict_file = "dict.txt"
            stop_word_model = "stop_word_model.txt"
            self.tokenizer = jieba.Tokenizer(dict_file)
            self.stop_word_model = set(
                [line.strip() for line in open(stop_word_model, "r", encoding="utf-8").readlines()] + [" "]
            )
        # 读取语料库
        # corpus_df = pd.read_csv("corpus.csv")
        corpus_df = pd.read_csv(corpus_path)
        corpus_df["id"] = list(range(corpus_df.shape[0]))
        corpus_df = corpus_df.drop_duplicates(subset=["type", "corpus"]).reset_index()
        print(corpus_df.head())
        # 读取流程列表
        procs = pd.read_csv("procs.csv")
        procs["in_node"] = procs["in_node"].astype(np.str)

        # 读取流程管理词典，用于获取当前 pid - innode 下可用的 意图与对应的语料id
        self.dm_dct = {}
        proc_group = procs.groupby("processid")
        for pid, grp in proc_group:
            self.dm_dct[pid] = {}
            for in_node, gp in grp.groupby("in_node"):
                self.dm_dct[pid][in_node] = []
                intents = []
                for i, row in gp.iterrows():
                    intents.extend(row.semantic_type.split("+"))
                # print(pid, in_node, intents)
                idx, input_mask = get_intents_index_and_input_mask(intents, corpus_df)
                self.dm_dct[pid][in_node] = CorpusID(intents=intents, index=idx, input_mask=input_mask)
                # for st in row.semantic_type.split('+'):
                #     self.dm_dct[pid][in_node].append(st)

        # 处理语料，填充 self.index2corpus、self.intent2id、self.corpus_matrix
        self.bc = BertClient()
        if self.tokenizer:
            corpus_list = [self.normalize_sentence(corpus) for corpus in corpus_df.corpus.tolist()]
        else:
            corpus_list = [corpus[:60] if len(corpus) > 60 else corpus for corpus in corpus_df.corpus.tolist()]
        print("\n".join(corpus_list[:10]))
        self.corpus_matrix = self.bc.encode(corpus_list)
        self.corpus_matrix_norm = np.linalg.norm(self.corpus_matrix, axis=1)
        for i, row in corpus_df.iterrows():
            intent = row.type
            # if self.tokenizer:
            #     sentence = self.normalize_sentence(row.corpus, stop=False)
            # else:
            sentence = row.corpus

            weights = None
            self.index2corpus.append(Corpus(sentence, weights, intent))
            self.intent2id[intent].append(i)
        self.corpus_size = len(self.corpus_matrix)

    def tokenize(self, sentence, HMM=False, stop=True):
        """Parse input string into a list of tokens."""
        # 调用jieba分词
        tks = self.tokenizer.lcut(sentence, HMM=HMM)
        # 去停用词
        if tks == [" "]:
            tks = sentence
        if stop:
            tks_ = [x for x in tks if x not in self.stop_word_model]
        else:
            return tks
        if len(tks_) == 0:
            return tks
        else:
            return tks_

    def normalize(self, str_list):
        new_list = []
        str_list.append("#")
        for i in range(len(str_list) - 1):
            if str_list[i] != str_list[i + 1]:
                new_list.append(str_list[i])
        return new_list

    def normalize_sentence(self, sentence, stop=True):
        res = "".join(self.normalize(self.tokenize(sentence, stop=stop)))
        res_ = res if len(res) < 60 else res[:60]
        return res_

    def match(self, pid, innode, vector=None, scores=None, threshold=0.87, out_true=None, neg=5):
        """Return most similar Corpus."""
        # vector = self.bc.encode([sentence])[0]
        # 计算相似度
        try:
            candidates_intent = self.dm_dct[pid][innode]
        except KeyError:
            raise ProcessIdNotFoundError(pid)
        # TODO: candidates_index 和 input_mask 加入初始化的过程中
        # 有哪些语料对应的index需要保留
        candidates_index = candidates_intent.index
        # m = self.corpus_matrix[candidates_index]
        # score = np.dot(self.corpus_matrix[candidates_index], vector) / (
        #     np.linalg.norm(vector) * self.corpus_matrix_norm[candidates_index]
        # )
        if scores is None:
            score = np.sum(vector * self.corpus_matrix[candidates_index], axis=1) / (
                np.linalg.norm(vector) * np.linalg.norm(self.corpus_matrix[candidates_index], axis=1),
            )
        else:
            score = scores[candidates_index]
        # print(np.sort(score)[0, ::-1][:10])
        similarities_index = np.vstack((candidates_index, score)).T
        # similarities_index = np.array(list(zip(candidates_index, score.tolist())))
        # similarities_index = np.array(list(zip(candidates_index, score[0].tolist())))
        # 取相似度大于 threshold 的
        index_k = similarities_index[:, 1] >= threshold
        id2corpus_k = similarities_index[index_k][:, 0].astype(int)
        # sim = similarities_index[index_k][:, 1]
        corpus = []
        if len(id2corpus_k) < 10:
            # sim = []
            index_10 = np.argsort(similarities_index[:, 1])
            for i in index_10[-10:]:
                corpus.append(self.index2corpus[int(similarities_index[i][0])])
                # sim.append(similarities_index[i][1])
            # return corpus, bert_tokens, sim
        else:
            index_50 = np.argsort(similarities_index[index_k][:, 1])[-20:]
            id2corpus_k_new = similarities_index[index_k][index_50, 0]
            corpus = [self.index2corpus[int(index)] for index in id2corpus_k_new]
            # return corpus, bert_tokens, sim
        intents = [i.intent for i in corpus]
        outnodes = [self.dm.predict(pid, innode, i) for i in intents]
        corpuses = [i.sentence for i in corpus]
        # print("out_nodes", outnodes)
        # print("intens", intents)
        # print("corpus", corpuses)
        return intents, outnodes, corpuses

    def to_json(self, corpus, tokens, similarity):
        return {
            "corpus": corpus.sentence,
            "token": tokens,
            "sim_token": corpus.weights,
            "intent": corpus.intent,
            "similarity": similarity,
        }


if __name__ == "__main__":
    db_param = {
        "host": "192.168.162.192",
        "user": "robot_collection",
        "password": "robot_collection_20180523",
        "database": "robot",
        "corpus_table": "corpus_test",
    }
    a = CorpusBert()
    b = a.match("benrenshoucui", "1.1", "你搞错了")
    print(b)
