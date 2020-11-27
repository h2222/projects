# 使用Bert粗排生成triplet训练集
# from corpus_db import CorpusDB
from corpus_Bert import CorpusBert
import pandas as pd
import numpy as np
import os
import time
from sklearn.utils import shuffle
from bert_serving.client import BertClient


data_dir = "/home/zhaoxi.li/bert/data"
seed = 2019


def generate_pairs(cdb, msg_df, bc=None, threshold=0.87, train=True):
    if train:
        columns = ["processid", "in_node", "msg", "corpus_pos", "corpus_neg", "msg_origin"]
    else:
        columns = [
            "processid",
            "in_node",
            "msg",
            "corpus",
            "out_node",
            "out_true",
            "msg_type",
            "corpus_type",
            "target",
            "msg_origin",
        ]

    total_df = pd.DataFrame(columns=columns)

    t1 = time.time()
    df_list = []
    if cdb.tokenizer:
        msgs = [cdb.normalize_sentence(msg, stop=False) for msg in msg_df.msg.tolist()]
    else:
        msgs = [msg[:60] if len(msg) > 60 else msg for msg in msg_df.msg.tolist()]
    print("\n".join(msgs[:10]))
    for i, row in msg_df.iterrows():
        # if i == 11:
        #    break
        if i % 1000 == 0:
            t2 = time.time()
            if bc:
                if i + 1000 <= len(msgs):
                    msg_vectors = bc.encode(msgs[i : i + 1000])
                else:
                    msg_vectors = bc.encode(msgs[i:])
                scores = np.dot(msg_vectors, cdb.corpus_matrix.T) / (
                    np.linalg.norm(msg_vectors, axis=1).reshape(-1, 1) * cdb.corpus_matrix_norm
                )
            t3 = time.time()
            # tmp_df['target'] = tmp_df.apply(lambda l : int(l['out_true'] == l['out_node']), axis = 1)
            # idx = tmp_df.groupby(['processid', 'in_node', 'msg', 'corpus', 'out_true'])['target'].idxmax()
            # total_df = total_df.append(tmp_df.loc[idx])
            if i > 0:
                tmp_df = pd.concat(df_list, axis=0, ignore_index=True)
                if not train:
                    tmp_df["target"] = tmp_df["out_true"].eq(tmp_df["out_node"]).map({True: 1, False: 0})
                total_df = total_df.append(tmp_df[columns])
                df_list = []
                print("Sample %d: %.2f, %.2f, %d" % (i, t3 - t1, t3 - t2, total_df.shape[0]))
        #    break
        if bc:
            intents, outnodes, corpuses = cdb.match(
                row.processid, row.in_node, scores=scores[i % 1000], threshold=threshold
            )
        else:
            intents, outnodes, corpuses = cdb.match(row.processid, row.in_node, row.msg)
        if train:
            pos_indexs = []
            neg_indexs = []
            for i, outnode in enumerate(outnodes):
                if outnode == row.out_true:
                    pos_indexs.append(i)
                else:
                    neg_indexs.append(i)
            # print(len(pos_indexs), len(neg_indexs))
            corpus_pos_list = []
            corpus_neg_list = []
            if row.out_true != "-99":
                for pos in pos_indexs:
                    for neg in neg_indexs:
                        corpus_pos_list.append(corpuses[pos])
                        corpus_neg_list.append(corpuses[neg])

            else:
                for neg in neg_indexs:
                    corpus_pos_list.append(row.msg)
                    corpus_neg_list.append(corpuses[neg])
            df = pd.DataFrame(
                {
                    "processid": row.processid,
                    "in_node": row.in_node,
                    "msg": row.msg,
                    "corpus_pos": corpus_pos_list,
                    "corpus_neg": corpus_neg_list,
                    "msg_origin": row.msg,
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "processid": row.processid,
                    "in_node": row.in_node,
                    "msg": row.msg,
                    "corpus": corpuses,
                    "out_node": outnodes,
                    "out_true": row.out_true,
                    "msg_type": row.type,
                    "corpus_type": intents,
                    "msg_origin": row.msg,
                }
            )
        # print(row.msg, msgs[i])
        # "msg": msgs[i],
        # "msg": row.msg
        # df = df.drop_duplicates(subset=['out_node', 'corpus'], keep='first', inplace=False)
        #         df['target'] = df.apply(lambda l : int(l['out_true'] == l['out_node']), axis = 1)
        # idx = df.groupby('corpus')['target'].idxmax()
        # df_list.append(df.loc[idx])
        df_list.append(df)
        # total_df = total_df.append(df.loc[idx].reset_index())
    if df_list:
        tmp_df = pd.concat(df_list, axis=0, ignore_index=True)
        if not train:
            tmp_df["target"] = tmp_df["out_true"].eq(tmp_df["out_node"]).map({True: 1, False: 0})
        total_df = total_df.append(tmp_df[columns])
    total_df = total_df.reset_index()
    total_df = total_df[columns]
    # idx = total_df.groupby(['processid', 'in_node', 'msg', 'corpus', 'out_true'])['target'].idxmax()
    # total_df = total_df.loc[idx]
    #     total_df = pd.concat(df_list, axis=0, ignore_index=True)
    total_df = shuffle(total_df, random_state=seed)
    print(total_df.head())
    print(total_df.shape)
    return total_df


if __name__ == "__main__":
    # cdb = CorpusDB()
    # bc = None
    cdb = CorpusBert(os.path.join(data_dir, "corpus.csv"))
    # cdb = CorpusBert(os.path.join(data_dir, "new_corpus_tokenize6.csv"), tokenizer=True)
    bc = BertClient()
    # cdb = CorpusBert()
    # 生产样本之前需要先启动Bert-as-service的server端
    # bert-serving-start -model_dir /home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/ -num_worker=6 -device_map 1 -max_seq_len 64 -max_batch_size 16 -pooling_layer -1
    train_msg_path = os.path.join(data_dir, "msg.csv")
    test_msg_path = os.path.join(data_dir, "test_20000.csv")
    train_msg = pd.read_csv(train_msg_path)
    train_msg = train_msg.drop(np.where(train_msg.msg == " ")[0], axis=0).reset_index()
    train_msg["in_node"] = train_msg["in_node"].astype(np.str)
    train_df = generate_pairs(cdb, train_msg, bc, threshold=0.87, train=True)
    train_df.to_csv(os.path.join(data_dir, "train_bert_87_triplet_cut20+invalid1.csv"), index=None)
    test_msg = pd.read_csv(test_msg_path)
    test_msg = test_msg.drop(np.where(test_msg.msg == " ")[0], axis=0).reset_index()
    test_msg["in_node"] = test_msg["in_node"].astype(np.str)
    test_df = generate_pairs(cdb, test_msg, bc, threshold=0.87, train=False)
    test_df.to_csv(os.path.join(data_dir, "test_bert_87_triplet_cut201.csv"), index=None)
