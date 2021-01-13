# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import random
from tqdm import tqdm
import pickle
import json
import time
from sklearn.metrics import *
import numpy as np
import sys
from consts import *
from util import *
import redis

batch_size = 512
embed_size = 8
epochs = 20
lam = 0.01
lr = 0.01


class TrainData:
    def __init__(self, users, preferred_items):
        self.users = users
        self.preferred_items = preferred_items
        assert len(users) == len(preferred_items)
        self.n = len(users)
        self.u_i_set = {}
        self.max_item_id = -1
        for i in range(self.n):
            uid = self.users[i]
            iid = self.preferred_items[i]
            self.max_item_id = max(self.max_item_id, iid)
            if uid not in self.u_i_set:
                self.u_i_set[uid] = {}
            self.u_i_set[uid][iid] = True
    def get_data(self):
        global batch_size
        users = []
        items = []
        other_items = []
        while len(users) < batch_size:
            i = random.randint(0, self.n - 1)
            uid = self.users[i]
            users.append(uid)
            items.append(self.preferred_items[i])
            while True:
                x = random.randint(0, self.max_item_id)
                if x not in self.u_i_set[uid]:
                    other_items.append(x)
                    break
        return users, items, other_items
    def get_num_batches(self):
        return max(self.n - 1, 0) / batch_size + 1


class Model:
    def __init__(self, user_size, item_size, embed_size, num_batches, learning_rate=0.01, lambd=0.01):
        self.graph = tf.Graph()
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = embed_size
        self.num_batches = num_batches
        self.lambd = lambd
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.build_graph()
    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.train_users = tf.placeholder(tf.int32, name='train_users')
            self.train_preferred_items = tf.placeholder(tf.int32, name='train_preferred_items')
            self.train_other_items = tf.placeholder(tf.int32, name='train_other_items')
    def _create_embedding(self):
        with tf.name_scope("embed"):
            self.U = tf.get_variable("U", shape=[self.user_size, self.embed_size],
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=1))
            self.I = tf.get_variable("I", shape=[self.item_size, self.embed_size],
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.lambd),
                                     initializer=tf.contrib.layers.xavier_initializer(seed=1))
    def _calc_latent_score(self, users, items):
        user_embed = tf.nn.embedding_lookup(self.U, users)
        reshaped_user_embed = tf.reshape(user_embed, [-1, 1, self.embed_size])
        item_embed = tf.nn.embedding_lookup(self.I, items)
        reshaped_item_embed = tf.reshape(item_embed, [-1, self.embed_size, 1])  # transposed
        latent_score = tf.squeeze(tf.matmul(reshaped_user_embed, reshaped_item_embed))
        return latent_score
    def _create_loss(self):
        with tf.name_scope("loss"):
            preferred_score = self._calc_latent_score(self.train_users, self.train_preferred_items)
            other_score = self._calc_latent_score(self.train_users, self.train_other_items)
            diff = preferred_score - other_score
            self.diff = tf.reduce_sum(- tf.sigmoid(diff), name='diff')
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
            self.loss_with_reg = tf.add(self.diff, self.reg_loss, name='loss_with_reg')
    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_with_reg)
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("diff", self.diff)
            tf.summary.scalar("reg_loss", self.reg_loss)
            tf.summary.scalar("loss_with_reg", self.loss_with_reg)
            self.summary_op = tf.summary.merge_all()
    def build_graph(self):
        with self.graph.as_default() as g:
            with g.device('/cpu:0'):
                self._create_placeholders()
                self._create_embedding()
                self._create_loss()
                self._create_optimizer()
                self._create_summaries()
    def train(self, train_data, model_dir, num_train_steps=10):
        os.system('rm -rf ' + model_dir)
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(model_dir + 'board', sess.graph)
            step = 0
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            for batch in tqdm(range(num_train_steps * self.num_batches)):
                batch_users, batch_items, other_items = train_data.get_data()
                feed_dict = {self.train_users: batch_users,
                             self.train_preferred_items: batch_items,
                             self.train_other_items: other_items}
                _, summary = sess.run([self.optimizer, self.summary_op], feed_dict=feed_dict)
                step += 1
                if step % 100 == 0:
                    writer.add_summary(summary, global_step=step)
                    writer.flush()
                    saver.save(sess, model_dir + 'model', global_step=step)
            saver.save(sess, model_dir + 'model', global_step=step)


def get_data(filePath):
    user_dict = {}
    item_dict = {}
    data_file = open(filePath)
    users = []
    preferred_items = []
    for line in data_file:
        try:
            rec = json.loads(line)
            uid = int(rec['user_id'])
            iid = int(rec['good_id'])
            cn = int(rec['click_num'])
            if cn > 0:
                if uid not in user_dict:
                    user_dict[uid] = 0
                if iid not in item_dict:
                    item_dict[iid] = 0
                users.append(uid)
                preferred_items.append(iid)
                user_dict[uid] += 1
                item_dict[iid] += 1
        except:
            pass
    data_file.close()
    user_tmp = zip(user_dict.keys(), user_dict.values())
    item_tmp = zip(item_dict.keys(), item_dict.values())
    print "all_user size: %d, all_item size: %d" % (len(user_tmp), len(item_tmp))
    user_tmp.sort(key=lambda x: x[1], reverse=True)
    item_tmp.sort(key=lambda x: x[1], reverse=True)
    user_tmp = [x for x in user_tmp if x[-1] >= min_user_active]
    print "cut_user size: %d" % len(user_tmp)
    users_num = len(user_tmp) + 1
    items_num = len(item_tmp) + 1
    new_user_dict = {}
    new_item_dict = {}
    for x in range(len(user_tmp)):
        new_user_dict[user_tmp[x][0]] = x
    for y in range(len(item_tmp)):
        new_item_dict[item_tmp[y][0]] = y
    users = [new_user_dict[x] for x in users]
    preferred_items = [new_item_dict[x] for x in preferred_items]
    train_data = TrainData(users, preferred_items)
    del users
    del preferred_items
    users_num += 1
    items_num += 1
    return train_data, users_num, items_num, new_user_dict, new_item_dict


def get_test_data(filePath):
    in_user = 0
    not_in_user = 0
    in_item = 0
    not_in_item = 0
    data_file = open(filePath)
    users = []
    preferred_items = []
    label = []
    min_ts = int(time.time()) - 86400
    for line in data_file:
        try:
            rec = json.loads(line)
            ts = int(rec["ts"])
            if ts <= min_ts:
                continue
            uid = int(rec['user_id'])
            iid = int(rec['good_id'])
            label.append(int(rec['label']))
            if uid in user_dict:
                in_user += 1
                users.append(user_dict[uid])
            else:
                not_in_user += 1
                users.append(-1)
            if iid in item_dict:
                in_item += 1
                preferred_items.append(item_dict[iid])
            else:
                not_in_item += 1
                preferred_items.append(-1)
        except:
            pass
    data_file.close()
    print in_user, not_in_user, in_item, not_in_item
    return users, preferred_items, label


def evaluate(yy, y, u):
    uy = {}
    uyy = {}
    for i in range(len(y)):
        if u[i] not in uy:
            uy[u[i]] = []
            uyy[u[i]] = []
        uy[u[i]].append(y[i])
        uyy[u[i]].append(yy[i])
    aauc = 0
    acnt = 0
    for uid in uy:
        s = sum(uy[uid])
        if 0 < s < len(uy[uid]) and len(uy[uid]) >= 5:
            acnt += 1
            aauc += roc_auc_score(uy[uid], uyy[uid])
    print roc_auc_score(y, yy)
    print aauc / acnt, acnt


def load_model():
    sess = tf.Session()
    checkpoint_file = tf.train.latest_checkpoint(model_dir)
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file), clear_devices=True)
    saver.restore(sess, checkpoint_file)
    graph = sess.graph
    g_u = graph.get_operation_by_name("U").outputs[0]
    g_i = graph.get_operation_by_name("I").outputs[0]
    t_u, t_i = sess.run([g_u, g_i])
    res_u = np.array(t_u)
    res_i = np.array(t_i)
    return res_u, res_i


def predict(test_u, test_i, label, user_positive_num=5):
    U, I = load_model()
    default_u = np.mean(U, axis=0).reshape((1, embed_size))
    default_i = np.mean(I, axis=0).reshape((1, embed_size))
    U = np.concatenate((U, default_u))
    I = np.concatenate((I, default_i))
    print U.shape, I.shape
    U_arr = U[list(test_u)]
    I_arr = I[list(test_i)]
    res = np.sum(np.multiply(U_arr, I_arr), axis=-1)
    y_true = label
    y_pred = list(res)
    user_res = {}
    for x in range(len(y_pred)):
        if test_u[x] not in user_res:
            user_res[test_u[x]] = {
                'score': [],
                'label': []
            }
        user_res[test_u[x]]['score'].append(y_pred[x])
        user_res[test_u[x]]['label'].append(y_true[x])
    user_count = 0
    user_auc = 0.0
    for user in user_res:
        try:
            if sum(user_res[user]['label']) > user_positive_num:
                user_count += 1
                user_auc += float(roc_auc_score(user_res[user]['label'], user_res[user]['score']))
        except:
            pass
    print "the amount of user: %d" % user_count
    print "the user average auc {}".format(user_auc / float(user_count))
    print "roc_auc_score:", roc_auc_score(y_true, y_pred)


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
    train, users_num, items_num, user_dict, item_dict = get_data(path1)
    print "Valid Ids size: %d" % len(valid_ids)
    print 'Train size: %d' % train.n
    print 'Users: %d' % users_num
    print 'Items: %d' % items_num
    model = Model(users_num, items_num, embed_size, train.get_num_batches())
    model.train(train, model_dir, epochs)
    U, I = load_model()
    # remove_data(dt, model_type)
    # rec goods 2 users
    rec_res = get_personas_goods(U, I, user_dict, item_dict, valid_ids=valid_ids, max_goods_num=100)
    # rec goods via similar goods
    item_rec_res = get_personas_goods(I, I, item_dict, item_dict, valid_ids=valid_ids, max_goods_num=50)
    #
    get_default_goods("bpr", "u2g", U, I, item_dict, valid_ids=valid_ids, max_goods_num=100)
    write_algo2rds(rec_res, "bpr", "u2g")
    write_algo2rds(item_rec_res, "bpr", "g2g")
    #
    # TODO do not need
    f_path = open("/data/omall_rec_%s.txt" % dt, "aw")
    for uid in rec_res:
        f_path.write(json.dumps({
            "algo": "bpr",
            "type": uid2goods,
            "user_id": uid,
            "goods_id": rec_res[uid],
        }))
        f_path.write("\n")
    for item_id in item_rec_res:
        f_path.write(json.dumps({
            "algo": "bpr",
            "type": goods2goods,
            "item_id": item_id,
            "goods_id": item_rec_res[item_id]
        }))
        f_path.write("\n")
    f_path.close()