#coding=utf-8
# 二分类  识别工单中，真实未上车
import tensorflow as tf
import numpy as np
import os, sys
from gensim.models import KeyedVectors
from tensorflow.python.framework import graph_util
from datetime import datetime
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk import tokenize

sw = stopwords.words('portuguese')


TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

# GPU 选择
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        print("************ epach:%d ************" % epoch)
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_text_dataset(path1, path2):
    pos_df = pd.read_csv(path1, sep='|', encoding='utf-8')
    neg_df = pd.read_csv(path2, sep='|', encoding='utf-8')
    pos_df.columns = ["ticket_id", "country_code", "organization_id", "content", "trans", "category_id"]
    neg_df.columns = ["ticket_id", "country_code", "organization_id", "content", "trans", "category_id"]

    # cut size
    # pos_df = pos_df.loc[:200, :]
    # neg_df = pos_df.loc[:200, :]
    
    num_pos_train = pos_df.shape[0]
    num_neg_train = neg_df.shape[0]
    X = pos_df.append(neg_df, ignore_index=True)
    Y = np.zeros((num_pos_train + num_neg_train, 2), dtype=np.int)
    Y[:num_pos_train, 0] = 1
    Y[num_pos_train:, 1] = 1
    X_train, X_vali_test, Y_train, Y_vali_test = train_test_split(X, Y, test_size=0.003, random_state = 666)
    X_vali, X_test, Y_vali, Y_test = train_test_split(X_vali_test, Y_vali_test, test_size=0.2, random_state = 666)
    print(X_train.shape)
    print(Y_test.shape)
    print(X_vali.shape)
    print(Y_vali.shape)
    print(X_test.shape)
    print(Y_test.shape)
    print(X.shape)
    return X_train, Y_train, X_vali, Y_vali, X_test, Y_test


def data_process(text_batch, label_batch, sentence_len, w2v_vocab):
    res_content = []
    res_label = []
    for content, label in zip(text_batch, label_batch) :
        try:
            word_ls =  tokenize.word_tokenize(content, language='portuguese') 
            word_ls = [i for i in word_ls if (i not in sw) and (i in w2v_vocab)]
            while (len(word_ls) != sentence_len):
                if len(word_ls) > sentence_len:
                    word_ls = word_ls[:sentence_len]
                elif len(word_ls) < sentence_len:
                    word_ls.append("<PAD>")
            res_content.append(word_ls)
            res_label.append(label)
        except:
            continue
    res_content = np.array(res_content)
    return res_content, res_label


def flush_save_path(path):
    """
    清空文件夹
    """
    if os.path.exists(path):
        os.system('rm -rf %s' % path)
        os.mkdir(path)
    else:
        os.mkdir(path)
    return path


def run_train_step(model, sess, train_x, train_y, merged):
    """
    训练步骤
    """
    feed_dict = {model.input_sent : train_x, model.input_y : train_y}
    to_return = [model.train_op,
                 model.probabilitys, 
                 model.predictions,
                 model.loss,
                 model.global_step,
                 model.accuracy,
                 model.tpr,
                 model.precision,
                 merged]
    train_op, prob, pred, loss, gp, acc, tpr, pre, visual = sess.run(to_return, feed_dict)
    return train_op, prob, pred, loss, gp, acc, tpr, pre, visual


def run_eval_step(model, sess, vali_x, vali_y, merged):
    """
    验证步骤
    """
    feed_dict = {model.input_sent : vali_x, model.input_y : vali_y}
    to_return = [model.accuracy,
                 model.tpr,
                 model.precision,
                 model.global_step,
                 merged]
    acc, tpr, pcs, gp, visual = sess.run(to_return, feed_dict)
    return acc, tpr, pcs, gp, visual


class TEXTCNN:
    """
    Text CNN model
    """
    def __init__(self, batch_size, num_epochs, save_path, l2_alpha, dropout, sentence_len, attention, w2v_path, w2v_trainalbe):
        # 定义参数
        self.save_every = 50  # 每50次训练，在验证集上看效果
        self.sequence_length = sentence_len  # 句子长度
        self.word_dim = 105  # 给ai lab的请求参数
        self.filter_sizes = [3, 4, 5]  # 三个滤波器
        self.num_filters = 128  # 使用默认值就ok
        self.num_classes = 2  # 二分类问题
        self.embedding_size = self.word_dim  # embedding 的维度，即词向量的维度。直接由AI LAB提供。
        self.l2_reg_lambda = l2_alpha
        self.dropout_keep_prob = dropout #0.9  # droput_keep 的比率
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_path = save_path #flush_save_path('./model_simple_2')
        self.attention = attention
        self.w2v_path = w2v_path
        self.w2v_trainalbe = w2v_trainalbe

        # with tf.name_scope('Input'):
        # self.x_vec_v1 = tf.placeholder(tf.float32, [None, self.sequence_length * self.word_dim], name="Input_x")
        # xdim = tf.shape(self.x_vec_v1)

        self.input_sent = tf.placeholder(tf.string, [None, self.sequence_length], name="Input_sent")
        self.input_y = tf.placeholder(tf.double, [None, self.num_classes], name="Input_y")
        print("*" * 20)
        print('sequence length: {}\nword dimension:{}\nclass number:{}\nbatch size:{}\nepoch number:{}'\
        .format(self.sequence_length, self.word_dim, self.num_classes, self.batch_size, self.num_epochs))
        print("*" * 20)

        # loading w2v table
        w2v_model = KeyedVectors.load_word2vec_format(self.w2v_path)
        vec_dim = w2v_model.vector_size
        total_vocab_size = len(w2v_model.vocab)
        embedding_mat = np.zeros((total_vocab_size, vec_dim), dtype=np.double)
        self.sorted_vocab = sorted(w2v_model.vocab)
        vocab_tf = tf.constant(self.sorted_vocab)
        self.word_table = tf.contrib.lookup.index_table_from_tensor(mapping=vocab_tf, num_oov_buckets=1, default_value=-1)
        for idx, word in enumerate(self.sorted_vocab):
            embedding_mat[idx] = w2v_model.get_vector(word)
        self.word_embedding_table = tf.get_variable("embedd", [total_vocab_size, vec_dim], initializer = tf.constant_initializer(embedding_mat), trainable=self.w2v_trainalbe, dtype=tf.double)


    def build_graph(self):
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, dtype=tf.double)
        pooled_outputs = []

        # version 4:  trainable embedding table
        word_idx = self.word_table.lookup(self.input_sent)
        self.x_vec  = tf.nn.embedding_lookup(self.word_embedding_table, word_idx, name="Input_x_v2")
       
        # version3 + attention  word_dim=embedding_size=100, attention_hidden_dim =100
        if self.attention:
            self.attention_hidden_dim = self.word_dim
            self.attention_w = tf.Variable(
                tf.random_normal([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0, dtype=tf.double),
                name="attention_w", dtype=tf.double)
            self.attention_u = tf.Variable(
                tf.random_normal([self.embedding_size, self.attention_hidden_dim], 0.0, 1.0, dtype=tf.double),
                name="attention_u", dtype=tf.double)
            self.attention_v = tf.Variable(
                tf.random_normal([self.attention_hidden_dim, 1], 0.0, 1.0, dtype=tf.double),
                name="attention_v", dtype=tf.double)
            
            # attention layer before conv
            self.output_attn = []
            with tf.name_scope("attention"):
                input_att = tf.split(self.x_vec, self.sequence_length, axis=1)
                for index, x_i in enumerate(input_att):
                    x_i = tf.reshape(x_i, [-1, self.embedding_size]) # [b, word_dim]
                    c_i = self.attention_func(x_i, input_att, index) # [b, word_dim]
                    inp = tf.concat([x_i, c_i], axis=1) # [b, 2 * word_dim] 注意的tensor和没注意的tensor拼接
                    self.output_attn.append(inp)
                
                # [b, seq_len, 2 * word_dim] <= [b , seq_len * 2 * word_dim]
                self.input_x_temp = tf.reshape(tf.concat(self.output_attn, axis=1),
                                            [-1, self.sequence_length, 2 * self.embedding_size])

                # [b, seq_len, 2 * word_dim, 1] <= [b, seq_len, 2 * word_dim]
                self.input_x = tf.expand_dims(self.input_x_temp, -1)
                dim_input_conv = self.input_x.shape[-2].value
        else:
            self.input_x = tf.expand_dims(self.x_vec, -1)
            dim_input_conv = self.embedding_size

        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, dim_input_conv, 1, self.num_filters]
                # with tf.name_scope('weights'):
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, dtype=tf.double), name="W", dtype=tf.double)
                tf.summary.histogram(name=str(filter_size)+'/weight',values=W)
                # with tf.name_scope('biases'):
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters], dtype=tf.double), name="b", dtype=tf.double)
                tf.summary.histogram(name=str(filter_size)+'/biase', values=b)
                conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                print("hidden conv size: ", h.shape)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
                print("pooling result conv size: ", pooled.shape)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        print("total filter concat pooled result: ", h_pool.shape)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print("total filter concat pooled result (flatten): ", h_pool.shape)
        # Dropout
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        # 输出
        W = tf.get_variable("W", shape=[num_filters_total, self.num_classes], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.double)
        b = tf.Variable(tf.constant(0.1, shape=[self.num_classes], dtype=tf.double), name="b", dtype=tf.double)
        
        if l2_alpha != 0:
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
        
        # 非归一化的分值
        # with tf.name_scope('score'):
        scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        tf.summary.histogram(name='score', values=scores)
        # 预测概率和预测分类
        self.probabilitys = tf.nn.softmax(scores, name="Output")
        tf.summary.histogram(name='probability', values=self.probabilitys)
        self.predictions = tf.argmax(self.probabilitys, 1, name="predictions")
        # 评估方法
        self.classify_evaluation()
        #损失函数+正则化x
        # with tf.name_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
        tf.summary.scalar('loss', self.loss)
        # 优化器+梯度优化
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)


    def attention_func(self, x_i, x, index):
        """
        attention layer
        :param x_i: the embedded input at time i
        :param x: the embedded input of all times(x_j of attentions)
        :param index: step of time
        """
        e_i = [] # weight value list
        c_i = [] # weighted tensor list
        for output in x: # [b, 1, word_dim] list
            output = tf.reshape(output, [-1, self.embedding_size]) # [b * 1, word_dim ]
            # [b , att_dim]  <=  ([b, word_dim] * [word_dim, att_dim]) + ([b*1, word_dim] * [word_dim, att_dim])
            attn_hidden = tf.tanh(tf.add(tf.matmul(x_i, self.attention_w), tf.matmul(output, self.attention_u)))
            # [b, 1]  <= [b, att_dim] * [att_dim, 1]
            e_i_j = tf.matmul(attn_hidden, self.attention_v)
            # [[b, 1], [b, 1], ...]
            e_i.append(e_i_j)
        # [b, seq_len]
        e_i = tf.concat(e_i, axis=1)
        # alpha 权重
        alpha_i = tf.nn.softmax(e_i, axis=1)
        alpha_i = tf.split(alpha_i, self.sequence_length, axis=1)
        # i != j
        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = tf.reshape(output, [-1, self.embedding_size])
                c_i_j = tf.multiply(alpha_i_j, output) # 权重 * 其他词向量
                c_i.append(c_i_j)
        # [b, seq_len-1, word_dim] # 其他词对index词的注意力权重tensor做 concat
        c_i = tf.reshape(tf.concat(c_i, axis=1), [-1, self.sequence_length - 1, self.embedding_size])
        # [b, word_dim] <= [b, seq_len-1, word_dim] # 向量叠加
        c_i = tf.reduce_mean(c_i, 1)
        return c_i


    def classify_evaluation(self):
        actuals = tf.argmax(self.input_y, 1)
        correct_predictions = tf.equal(self.predictions, actuals)
        
        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(self.predictions)
        zeros_like_predictions = tf.zeros_like(self.predictions)

        # ture possitive
        tp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(self.predictions, zeros_like_predictions)),
                "float"))
        # false negative
        fn_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(self.predictions, ones_like_predictions)),
                "float"))
        # false possitive
        fp_op = tf.reduce_sum(
            tf.cast(
                tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(self.predictions, zeros_like_predictions)),
                "float"))
        
        # true postive rate
        self.tpr = tf.math.divide_no_nan(tp_op, tf.add(tp_op, fn_op))
        tf.summary.scalar('recall', self.tpr)
        # precision
        self.precision = tf.math.divide_no_nan(tp_op, tf.add(tp_op, fp_op))
        tf.summary.scalar('precision', self.precision)


if __name__ == "__main__":
    # 建图,初始化
    pos_path = sys.argv[1]
    neg_path = sys.argv[2]
    w2v_path = sys.argv[3]
    batch_sz = 68
    epoch = 2
    dropout_keep = 0.7
    l2_alpha = 0.3
    sentence_len = 30
    transvt = False
    attention = True
    pn_1vs1 = False
    w2v_trainalbe = True
    save_path = "model_simple_v4"
    train_param = {"acc":0, "recall":0, "precision":0, "ct":0}
    vali_param = {"acc":0, "recall":0, "precision":0, "ct":0}
    tf.set_random_seed(666)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    model = TEXTCNN(batch_size=batch_sz, 
                    dropout=dropout_keep, 
                    l2_alpha=l2_alpha, 
                    num_epochs=epoch,
                    sentence_len=sentence_len, 
                    attention=attention,
                    save_path=save_path,
                    w2v_path=w2v_path,
                    w2v_trainalbe=w2v_trainalbe)
    model.build_graph()
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
    merged = tf.summary.merge_all()
    visual_path = os.path.join('visualization',"single_model_bz-%d_e-%d_do-%.2f_l2-%.2f_wz-%d_transvt-%s_attn-%s_%s" % (
                                    model.batch_size, model.num_epochs, model.dropout_keep_prob, model.l2_reg_lambda, model.sequence_length, transvt, attention,TIMESTAMP))
    writer = tf.summary.FileWriter(visual_path, graph=sess.graph)
    
    # loading data
    X_train, Y_train, _X_vali, _Y_vali, X_test, Y_test = load_text_dataset(pos_path, neg_path)

    batches = batch_iter(list(zip(X_train["content"], Y_train)),
                        batch_size=model.batch_size,
                        num_epochs=model.num_epochs)
    
    # 训练与验证
    for batch in batches:
        # 训练
        x_batch, y_batch = zip(*batch)
        x_batch, y_batch = data_process(x_batch, y_batch, model.sequence_length, model.sorted_vocab) # 分词 + padding
        # print(len(x_batch), len(y_batch))
        _, _, _, loss, gp, acc, tpr, pcs, visual = run_train_step(model, sess, x_batch, y_batch, merged)
        print("train_step:{}, loss:{}, accuracy:{}, recall:{}, precision:{}".format(gp, loss, acc, tpr, pcs))
        train_param["acc"] += acc
        train_param["recall"] += tpr
        train_param["precision"] += pcs
        train_param["ct"] += 1
        writer.add_summary(visual, gp)
        # 验证
        if gp % model.save_every == 0:
            # X_vali, Y_vali = data_process(x_batch, y_batch, model.sequence_length, model.sorted_vocab) # 分词 + padding 
            acc, tpr, pcs, gp, visual = run_eval_step(model, sess, x_batch, y_batch, merged)
            print("****\n evaluation_step:{}, loss:{}, accuracy:{},tpr:{}, precision:{} \n****".format(
                gp, loss, acc, tpr, pcs))
            vali_param["acc"] += acc
            vali_param["recall"] += tpr
            vali_param["precision"] += pcs
            vali_param["ct"] += 1
        writer.add_summary(visual, gp)

    train_acc = train_param["acc"] / (train_param["ct"] + 1e-5)
    train_rc =  train_param["recall"] / (train_param["ct"] + 1e-5)
    train_pcs = train_param["precision"] / (train_param["ct"] + 1e-5)
    print("(ave-train) acc:{}, rc:{}, pcs:{}".format(train_acc, train_rc, train_pcs))
    
    vali_acc = vali_param["acc"] / (vali_param["ct"] + 1e-5)
    vali_rc = vali_param["recall"] / (vali_param["ct"] + 1e-5)
    vali_pcs = vali_param["precision"] / (vali_param["ct"] + 1e-5)
    print("(ave-vali) acc:{}, rc:{}, pcs:{}".format(vali_acc, vali_rc, vali_pcs))

    # 保存方法3, 将table 初始化和variable 一起保存
    save_model_file = os.path.join(model.save_path, 'single_model_bz-%d_e-%d_do-%.2f_l2-%.2f_wz-%d_transvt-%s_pn1v1-%s_attn-%s_w2v5-%s_double_stable.pb' % (
                                    model.batch_size, model.num_epochs, model.dropout_keep_prob, model.l2_reg_lambda, model.sequence_length, transvt, pn_1vs1, attention, w2v_trainalbe))
    
    out_names = ['Output']
    for table_init_op in tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS):
            out_names.append(table_init_op.name)
    frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, out_names)
    tf.train.export_meta_graph(filename=save_model_file,
                               graph_def=frozen_graph,
                               collection_list=[tf.GraphKeys.TABLE_INITIALIZERS])

    # 模型保存2(保存模型文件夹,用于tf_serving的签名保存方法)
    # x = tf.get_default_graph().get_tensor_by_name('Input/Input:0')
    # y = tf.get_default_graph().get_tensor_by_name('predictions:0')
    # builder = tf.saved_model.builder.SavedModelBuilder(model.save_path)
    # signature = tf.saved_model.predict_signature_def(inputs={'input': x}, outputs={'output': y})
    # builder.add_meta_graph_and_variables(sess=sess, tags=['serve'], signature_def_map={'predict': signature})
    # builder.save()

    # with open(os.path.join(save_path, 'recode.txt'), 'a') as f:
    #     f.write("single_model_bz-%d_e-%d_do-%.2f_l2-%.2f_wz-%d_transvt-%s_attn-%s_newd_%s\n" % (
    #                                 model.batch_size, model.num_epochs, model.dropout_keep_prob, model.l2_reg_lambda, model.sequence_length, transvt, attention, TIMESTAMP))
    #     f.write("train:\n acc: %.2f rc: %.2f pcs: %.2f\n" % (train_acc, train_rc, train_pcs))
    #     f.write("vali:\n acc: %.2f rc: %.2f pcs: %.2f\n" % (vali_acc, vali_rc, vali_pcs))
        
