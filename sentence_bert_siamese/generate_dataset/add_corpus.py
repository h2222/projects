from dm import DM_Module
from collections import defaultdict
import pandas as pd
import numpy as np
import os
from bert_serving.client import BertClient
from datetime import datetime as datetime
from corpus_Bert import CorpusBert


data_dir = "/home/zhaoxi.li/bert/data"
seed = 2019


class CorpusAdder(object):
    def __init__(self, tokenizer=False):
        self.index2corpus = []  # 保存句向量_id对应的语料
        self.intent2id = defaultdict(list)  # 保存不同意图对应的id
        self.corpus_matrix = []  # 保存句向量
        self.tokenizer = tokenizer
        # 读取语料库
        corpus_df = pd.read_csv("corpus.csv")
        corpus_df = corpus_df.drop_duplicates(subset=["type", "corpus"]).reset_index()

        # 读取流程管理词典，用于获取当前 pid - innode 下可用的 意图与对应的语料id
        self.node2ids = {}
        corpus_df = corpus_df[corpus_df.processid.notna()].reset_index()
        corpus_df["id"] = list(range(corpus_df.shape[0]))

        procs = pd.read_csv("procs.csv")
        procs["in_node"] = procs["in_node"].astype(np.str)
        proc_group = procs.groupby("processid")
        for pid, gp0 in proc_group:
            self.node2ids[pid] = {}
            for in_node, gp1 in gp0.groupby("in_node"):
                self.node2ids[pid][in_node] = {}
                for out_node, gp2 in gp1.groupby("out_node"):
                    intents = []
                    for i, row in gp2.iterrows():
                        intents.extend(row.semantic_type.split("+"))
                    for intent in intents:
                        self.intent2id[intent].append((pid, in_node, out_node))
                    ids = np.arange(corpus_df.shape[0])[corpus_df["type"].isin(intents)]
                    self.node2ids[pid][in_node][out_node] = ids.tolist()
        self.corpus_df = corpus_df[["id", "processid", "type", "corpus", "create_time"]]

        # 处理语料，填充 self.index2corpus、self.intent2id、self.corpus_matrix
        if self.tokenizer:
            self.cdb = CorpusBert(os.path.join(data_dir, "corpus.csv"), tokenizer=True)
            corpus_list = [self.cdb.normalize_sentence(corpus, stop=False) for corpus in corpus_df.corpus.tolist()]
        else:
            corpus_list = corpus_df.corpus.tolist()
        self.bc = BertClient()
        self.corpus_matrix = self.bc.encode(corpus_list)
        # np.save("corpus_matrix_tokenize_nostop.npy", self.corpus_matrix)
        # self.corpus_matrix = np.load("corpus_matrix_tokenize_nostop.npy")
        self.corpus_matrix_norm = np.linalg.norm(self.corpus_matrix, axis=1)
        self.corpus_size = len(self.corpus_matrix)
        self.origin_corpus_size = len(self.corpus_matrix)
        self.add_pids = []
        self.add_types = []
        self.add_corpuses = []

    def generate_add_df(self):
        add_df = pd.DataFrame(
            {
                "id": np.arange(self.origin_corpus_size, self.corpus_size),
                "processid": self.add_pids,
                "type": self.add_types,
                "corpus": self.add_corpuses,
                "create_time": datetime.now(),
            }
        )
        return add_df

    def add_corpuses_all(self, msg_df, threshold=0.85):
        msg_df["id"] = list(range(msg_df.shape[0]))
        # msgs = [msg[:60] if len(msg) > 60 else msg for msg in msg_df.msg.tolist()]
        # msg_vecs = self.bc.encode(msgs)
        # np.save("msg_vecs_tokenize.npy", msg_vecs)
        msg_vecs = np.load("msg_vecs.npy")
        scores = np.dot(msg_vecs, self.corpus_matrix.T) / (
            np.linalg.norm(msg_vecs, axis=1).reshape(-1, 1) * self.corpus_matrix_norm
        )
        left_list = []
        for i, row in msg_df.iterrows():
            if i % 10000 == 0 and i > 0:
                print(f"Finish {i} of {msg_df.shape[0]}")
            match_ids = self.node2ids[row.processid][row.in_node].get(row.out_true)
            if match_ids is not None and len(match_ids) != 0:
                match_scores = scores[i, match_ids]
                if match_scores.max() < threshold:
                    self.add_corpus(msg_vecs[i], row)
            elif row.type in ["询问车贷"]:
                # self.add_corpus(msg_vecs[i], row)
                left_list.append(row.to_frame().T)
            elif row.type in ["新知识库", "新敏感词", "新节点"]:
                left_list.append(row.to_frame().T)
            else:
                left_list.append(row.to_frame().T)
        add_df = self.generate_add_df()
        left_df = pd.concat(left_list, axis=0)
        print(add_df.head())
        print(left_df.head())
        return add_df, left_df

    def add_corpus(self, msg_vec, row, increment=False):
        self.add_pids.append(row.processid)
        self.add_types.append(row.type)
        self.add_corpuses.append(row.msg)
        if increment:
            self.corpus_matrix = np.vstack((self.corpus_matrix, msg_vec))
            self.corpus_matrix_norm = np.hstack((self.corpus_matrix_norm, np.linalg.norm(msg_vec)))
            for pid, in_node, out_node in self.intent2id[row.type]:
                self.node2ids[pid][in_node][out_node].append(self.corpus_size)
        self.corpus_size += 1

    # 增量算法
    def add_corpuses_increment(self, msg_df, threshold=0.85):
        msg_df["id"] = list(range(msg_df.shape[0]))
        # msgs = [msg[:60] if len(msg) > 60 else msg for msg in msg_df.msg.tolist()]
        msg_vecs = np.load("msg_vecs.npy")
        msg_vecs_norm = np.linalg.norm(msg_vecs, axis=1)
        left_list = []
        for i, row in msg_df.iterrows():
            if i % 10000 == 0 and i > 0:
                print(f"Finish {i} of {msg_df.shape[0]}")
            scores = np.dot(self.corpus_matrix, msg_vecs[i]) / (msg_vecs_norm[i] * self.corpus_matrix_norm)
            match_ids = self.node2ids[row.processid][row.in_node].get(row.out_true)
            if match_ids is not None and len(match_ids) != 0:
                match_scores = scores[match_ids]
                if match_scores.max() < threshold:
                    self.add_corpus(msg_vecs[i], row, increment=True)
            elif row.type in ["询问车贷"]:
                self.add_corpus(msg_vecs[i], row, increment=True)
                left_list.append(row.to_frame().T)
                # pass
            elif row.type in ["新知识库", "新敏感词", "新节点"]:
                left_list.append(row.to_frame().T)
            else:
                left_list.append(row.to_frame().T)
        add_df = self.generate_add_df()
        left_df = pd.concat(left_list, axis=0)
        print(add_df.head())
        print(left_df.head())
        return add_df, left_df

    # 贪心算法
    def add_corpuses_greedy(self, msg_df, threshold=0.85, add_num=100):
        msg_df["id"] = list(range(msg_df.shape[0]))
        # msgs = [msg[:60] if len(msg) > 60 else msg for msg in msg_df.msg.tolist()]
        msg_vecs = np.load("msg_vecs.npy")
        msg_vecs_norm = np.linalg.norm(msg_vecs, axis=1)
        scores = np.dot(msg_vecs, self.corpus_matrix.T) / (msg_vecs_norm.reshape(-1, 1) * self.corpus_matrix_norm)
        left_list = []
        not_match_list = []
        for i, row in msg_df.iterrows():
            if i % 10000 == 0 and i > 0:
                print(f"Finish {i} of {msg_df.shape[0]}")
            match_ids = self.node2ids[row.processid][row.in_node].get(row.out_true)
            if match_ids is not None and len(match_ids) != 0:
                match_scores = scores[i, match_ids]
                if match_scores.max() < threshold:
                    not_match_list.append(i)
                    # self.add_corpus(msg_vecs[i], row, increment=True)
            elif row.type in ["询问车贷"]:
                not_match_list.append(i)
                # self.add_corpus(msg_vecs[i], row, increment=True)
                # left_list.append(row.to_frame().T)
                # pass
            elif row.type in ["新知识库", "新敏感词", "新节点"]:
                left_list.append(row.to_frame().T)
            else:
                left_list.append(row.to_frame().T)
        not_match_vectors = msg_vecs[not_match_list]
        not_match_scores = np.dot(not_match_vectors, not_match_vectors.T) / (
            msg_vecs_norm[not_match_list].reshape(-1, 1) * msg_vecs_norm[not_match_list]
        )
        not_match_df = msg_df.loc[not_match_list].copy().reset_index()
        not_match_df["id"] = np.arange(not_match_df.shape[0])
        same_in_out_mask = np.zeros_like(not_match_scores)
        for pid, gp0 in not_match_df.groupby("processid"):
            for in_node, gp1 in gp0.groupby("in_node"):
                for out_node, gp2 in gp1.groupby("out_true"):
                    tmp_ids = gp2["id"].tolist()
                    for tmp_id in tmp_ids:
                        same_in_out_mask[tmp_id, tmp_ids] = 1
        not_match_scores = np.multiply(not_match_scores, same_in_out_mask)
        for i in range(add_num):
            simi_flag_matrix = (not_match_scores > threshold).astype(np.int)
            index_with_max_simi = np.argmax(np.sum(simi_flag_matrix, axis=1), axis=0)
            tmp_add_list = np.where(simi_flag_matrix[index_with_max_simi] == 1)[0]
            print(i, index_with_max_simi, tmp_add_list.shape[0])
            not_match_scores[tmp_add_list] = 0
            not_match_scores[:, tmp_add_list] = 0
            index = not_match_list[index_with_max_simi]
            self.add_corpus(msg_vecs[index], msg_df.loc[index], increment=True)

        add_df = self.generate_add_df()
        left_df = pd.concat(left_list, axis=0)
        return add_df, left_df

    # 加无效语义惩罚的贪心算法
    def add_corpuses_greedy_reg(self, msg_df, threshold=0.85, add_num=100):
        invalid_index = msg_df.out_true == "-99"
        msg_df_valid = msg_df[~invalid_index].copy().reset_index()
        msg_df_invalid = msg_df[invalid_index].copy().reset_index()
        print(msg_df_valid.shape[0], msg_df_invalid.shape[0])
        if self.tokenizer:
            msgs = [self.cdb.normalize_sentence(msg, stop=False) for msg in msg_df_valid.msg.tolist()]
            msg_vecs = self.bc.encode(msgs)
            msgs_invalid = [self.cdb.normalize_sentence(msg, stop=False) for msg in msg_df_invalid.msg.tolist()]
            msg_vecs_invalid = self.bc.encode(msgs_invalid)
            # np.save("msg_vecs_tokenize_nostop.npy", msg_vecs)
            # np.save("msg_vecs_invalid_tokenize_nostop.npy", msg_vecs_invalid)
            # msg_vecs = np.load("msg_vecs_tokenize_nostop.npy")
            # msg_vecs_invalid = np.load("msg_vecs_invalid_tokenize_nostop.npy")
        else:
            msgs = [msg[:60] if len(msg) > 60 else msg for msg in msg_df.msg.tolist()]
            msg_vecs = self.bc.encode(msgs)
            msgs_invalid = [msg[:60] if len(msg) > 60 else msg for msg in msg_df_invalid.msg.tolist()]
            msg_vecs_invalid = self.bc.encode(msgs_invalid)
            # np.save("msg_vecs.npy", msg_vecs)
            # msg_vecs = np.load("msg_vecs.npy")
            # np.save("msg_vecs_invalid.npy", msg_vecs_invalid)
            # msg_vecs_invalid = np.load("msg_vecs_invalid.npy")
        msg_vecs_norm = np.linalg.norm(msg_vecs, axis=1)
        msg_vecs_norm_invalid = np.linalg.norm(msg_vecs_invalid, axis=1)
        scores = np.dot(msg_vecs, self.corpus_matrix.T) / (msg_vecs_norm.reshape(-1, 1) * self.corpus_matrix_norm)
        left_list = []
        not_match_list = []
        for i, row in msg_df_valid.iterrows():
            if i % 10000 == 0 and i > 0:
                print(f"Finish {i} of {msg_df_valid.shape[0]}")
            match_ids = self.node2ids[row.processid][row.in_node].get(row.out_true)
            if match_ids is not None and len(match_ids) != 0:
                match_scores = scores[i, match_ids]
                if match_scores.max() < threshold:
                    not_match_list.append(i)
                    # self.add_corpus(msg_vecs[i], row, increment=True)
            elif row.type in ["询问车贷"]:
                not_match_list.append(i)
                # self.add_corpus(msg_vecs[i], row, increment=True)
                # left_list.append(row.to_frame().T)
                # pass
            elif row.type in ["新知识库", "新敏感词", "新节点"]:
                left_list.append(row.to_frame().T)
            else:
                left_list.append(row.to_frame().T)
        print("%d valid messages, %d not match, %d left" % (msg_df_valid.shape[0], len(not_match_list), len(left_list)))
        not_match_vectors = msg_vecs[not_match_list]
        not_match_scores = np.dot(not_match_vectors, not_match_vectors.T) / (
            msg_vecs_norm[not_match_list].reshape(-1, 1) * msg_vecs_norm[not_match_list]
        )
        # not_match_vectors_invalid = msg_vecs_invalid[not_match_list_invalid]
        not_match_scores_invalid = np.dot(not_match_vectors, msg_vecs_invalid.T) / (
            msg_vecs_norm[not_match_list].reshape(-1, 1) * msg_vecs_norm_invalid
        )
        print(not_match_scores.shape, not_match_scores_invalid.shape)
        not_match_df = msg_df_valid.loc[not_match_list].copy()
        not_match_df["id"] = np.arange(not_match_df.shape[0])
        not_match_df_invalid = msg_df_invalid.copy()
        not_match_df_invalid["id"] = np.arange(not_match_df_invalid.shape[0])
        same_in_out_mask = np.zeros_like(not_match_scores)
        same_in_mask_invalid = np.zeros_like(not_match_scores_invalid)
        for pid, gp0 in not_match_df.groupby("processid"):
            for in_node, gp1 in gp0.groupby("in_node"):
                tmp_ids_invalid = not_match_df_invalid.loc[
                    (not_match_df_invalid.processid == pid) & (not_match_df_invalid.in_node == in_node), "id",
                ].tolist()
                # print(tmp_ids_invalid)
                for out_node, gp2 in gp1.groupby("out_true"):
                    if out_node == "-99":
                        continue
                    tmp_ids = gp2["id"].tolist()
                    for tmp_id in tmp_ids:
                        same_in_out_mask[tmp_id, tmp_ids] = 1
                        same_in_mask_invalid[tmp_id, tmp_ids_invalid] = 1
        not_match_scores = np.multiply(not_match_scores, same_in_out_mask)
        not_match_scores_invalid = np.multiply(not_match_scores_invalid, same_in_mask_invalid)
        simi_flag_matrix_invalid = not_match_scores_invalid > threshold
        b = np.sum(simi_flag_matrix_invalid, axis=1)
        for i in range(add_num):
            simi_flag_matrix = not_match_scores > threshold
            a = np.sum(simi_flag_matrix, axis=1)
            c = a / (b + 0.01)
            index_with_max_simi = np.argmax(c)
            tmp_add_list = np.where(simi_flag_matrix[index_with_max_simi] == 1)[0]
            not_match_scores[tmp_add_list] = 0
            not_match_scores[:, tmp_add_list] = 0
            index = not_match_list[index_with_max_simi]
            self.add_corpus(msg_vecs[index], msg_df_valid.loc[index], increment=True)

        add_df = self.generate_add_df()
        left_df = pd.concat(left_list, axis=0)
        print(add_df.head())
        print(left_df.head())
        return add_df, left_df


if __name__ == "__main__":
    ca = CorpusAdder(tokenizer=True)
    print("Init CorpusAdder finished.")
    data_dir = "/home/zhaoxi.li/bert/data"
    train_msg_path = os.path.join(data_dir, "msg.csv")
    test_msg_path = os.path.join(data_dir, "test_20000.csv")
    input_msg_df = pd.read_csv(train_msg_path)
    input_msg_df["in_node"] = input_msg_df["in_node"].astype(np.str)
    # input_msg_df = input_msg_df[input_msg_df.out_true != "-99"]
    input_msg_df = input_msg_df.drop(np.where(input_msg_df.msg == " ")[0], axis=0)
    input_msg_df = input_msg_df.drop_duplicates(subset=["processid", "in_node", "msg", "out_true"]).reset_index()
    print("Read input_msg_df finished.")
    # add_df, left_df = ca.add_corpuses_greedy(input_msg_df, threshold=0.85, add_num=500)
    add_df, left_df = ca.add_corpuses_greedy_reg(input_msg_df, threshold=0.85, add_num=500)
    # add_df, left_df = ca.add_corpuses_all(input_msg_df)
    # add_df, left_df = ca.add_corpuses_increment(input_msg_df)
    print(add_df.shape)
    add_df.to_csv(os.path.join(data_dir, "add_corpus_tokenize7.csv"), index=None)
    left_df.to_csv(os.path.join(data_dir, "left_corpus_tokenize7.csv"), index=None)
    new_df = ca.corpus_df.append(add_df)
    print(new_df.shape)
    new_df.to_csv(os.path.join(data_dir, "new_corpus_tokenize7.csv"), index=None)
