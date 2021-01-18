#coding=utf-8
# 二分类  识别工单中，真实未上车
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import graph_util
from datetime import datetime
import random
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
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
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
    feed_dict = {model.x_vec : train_x, model.input_y : train_y}
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
    feed_dict = {model.x_vec : vali_x, model.input_y : vali_y}
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
    def __init__(self, batch_size, num_epochs, save_path, l2_alpha, dropout, sentence_len):
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
        # with tf.name_scope('Input'):
        self.x_vec = tf.placeholder(tf.float32, [None, self.sequence_length * self.word_dim], name="Input_x")
        xdim = tf.shape(self.x_vec)
        self.input_x = tf.expand_dims(tf.reshape(self.x_vec, shape=[xdim[0], self.sequence_length, self.word_dim]), -1)
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="Input_y")
        print("*" * 20)
        print('sequence length: {}\nword dimension:{}\nclass number:{}\nbatch size:{}\nepoch number:{}'\
        .format(self.sequence_length, self.word_dim, self.num_classes, self.batch_size, self.num_epochs))
        print("*" * 20)

    def build_graph(self):
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                # with tf.name_scope('weights'):
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                tf.summary.histogram(name=str(filter_size)+'/weight',values=W)
                # with tf.name_scope('biases'):
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
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
        W = tf.get_variable("W", shape=[num_filters_total, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
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
        #损失函数+正则化
        # with tf.name_scope('loss'):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
        self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss
        tf.summary.scalar('loss', self.loss)
        # 优化器+梯度优化
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)


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
    batch_ls = [68]
    epoch_ls = [2]
    dropout_keep = 0.9
    l2_alpha = 0.0
    sentence_len = 30
    transvt = False
    pn_1vs1 = False
    save_path = "model_simple_1"
    train_param = {"acc":0, "recall":0, "precision":0, "ct":0}
    vali_param = {"acc":0, "recall":0, "precision":0, "ct":0}
    print(batch_ls, '\n', epoch_ls)
    for batch_sz in batch_ls:
        for epoch in epoch_ls:
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            model = TEXTCNN(batch_size=batch_sz, 
                            dropout=dropout_keep, 
                            l2_alpha=l2_alpha, 
                            num_epochs=epoch,
                            sentence_len=sentence_len, 
                            save_path=save_path)
            model.build_graph()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            merged = tf.summary.merge_all()
            visual_path = os.path.join('visualization',"single_model_bz-%d_e-%d_do-%.2f_l2-%.2f_wz-%d_transvt-%s_%s" % (
                                            model.batch_size, model.num_epochs, model.dropout_keep_prob, model.l2_reg_lambda, model.sequence_length, transvt, TIMESTAMP))
            writer = tf.summary.FileWriter(visual_path, graph=sess.graph)

            # 数据导入, batch
            train_ratio = 0.95
            # X_no_trip = np.load("test/no_trip_2020-09-19_09-22_req_time_2020-09-23_17_38_12.csv__len-"+str(sentence_len)+"_mat.npy")
            X_no_trip = np.load("test/data_4_5_9_10.csv_no_trip.csv__basew2v5_len-"+str(sentence_len)+"_mat.npy")
            X_trip = np.load("test/data_4_5_9_10.csv_others.csv__basew2v5_len-"+str(sentence_len)+"_mat.npy")
            num_no_trip = X_no_trip.shape[0]
            if pn_1vs1:
                # P:N = 1:1
                X_trip = X_trip[:num_no_trip, :]
            num_trip = X_trip.shape[0]
            
            print("no trip total:{}\ntrip total:{}".format(num_no_trip, num_trip))

            # train / vali_test      
            X_no_trip_train, X_no_trip_vali_test = np.split(X_no_trip, [int(num_no_trip * train_ratio)])
            X_trip_train, X_trip_vali_test = np.split(X_trip, [int(num_trip * train_ratio)])
            print("TRAIN\nno trip split point :{}\ntrip split point:{}".format(
                int(num_no_trip * train_ratio), int(num_trip * train_ratio)))
            # vali / test
            X_no_trip_vali, X_no_trip_test = np.split(X_no_trip_vali_test, [int(X_no_trip_vali_test.shape[0] * 0.5)])
            X_trip_vali, X_trip_test = np.split(X_trip_vali_test, [int(X_trip_vali_test.shape[0] * 0.5)])
            print("TEST\nno trip split point :{}\ntrip split point:{}".format(
                int(X_no_trip_vali_test.shape[0] * 0.5), int(X_trip_vali_test.shape[0] * 0.5)))
            
            num_pos_train = X_no_trip_train.shape[0]
            num_neg_train = X_trip_train.shape[0]
            num_pos_vali = X_no_trip_vali.shape[0]
            num_neg_vali = X_trip_vali.shape[0]
            num_pos_test = X_no_trip_test.shape[0]
            num_neg_test = X_trip_test.shape[0]

            print("train pos num: {}\ntrain neg num:{}\nvali pos num:{}\nvali neg num:{}\ntest pos num:{}\ntest neg num:{}".format(
                num_pos_train, num_neg_train, num_pos_vali, num_neg_vali, num_pos_test, num_neg_test))

            X_train = np.vstack((X_no_trip_train, X_trip_train))
            Y_train = np.zeros((num_pos_train + num_neg_train, 2), dtype=np.int)
            Y_train[:num_pos_train, 0] = 1
            Y_train[num_pos_train:, 1] = 1

            X_vali = np.vstack((X_no_trip_vali, X_trip_vali))
            Y_vali = np.zeros((num_pos_vali + num_neg_vali, 2), dtype=np.int)
            Y_vali[:num_pos_vali, 0] = 1
            Y_vali[num_pos_vali:, 1] = 1

            X_test = np.vstack((X_no_trip_test, X_trip_test))
            Y_test = np.zeros((num_pos_test + num_neg_test, 2), dtype=np.int)
            Y_test[:num_pos_test, 0] = 1
            Y_test[num_pos_test:, 1] = 1 

            if transvt:
                Y_test, Y_vali = Y_vali, Y_test
                X_test, X_vali = X_vali, X_test

            # save test data
            test_X_path = "test/X_test.npy"
            test_Y_path = "test/Y_test.npy"
            os.system('rm -rf %s %s' % (test_X_path, test_Y_path))
            X_test = np.save(test_X_path, X_test)    
            Y_test = np.save(test_Y_path, Y_test)  

            print("train X shape:{}\ntrain Y shape:{}\nVali X shape:{}\nVali Y shape:{}".format(
                X_train.shape, Y_train.shape, X_vali.shape, Y_vali.shape))


            # X_vali, X_train = np.split(X_total, [int(len(X_total) * 0.1)])
            # Y_train = np.zeros((len(X_train), 2), dtype=np.int)
            # Y_train[:len(X_train), 0] = 1
            # Y_vali = np.zeros([len(X_vali), 2])
            # Y_vali[:len(X_vali), 0] = 1
            batches = batch_iter(list(zip(X_train, Y_train)),
                                batch_size=model.batch_size,
                                num_epochs=model.num_epochs)
            
            # 训练与验证
            for batch in batches:
                # 训练
                x_batch, y_batch = zip(*batch)
                _, _, _, loss, gp, acc, tpr, pcs, visual = run_train_step(model, sess, x_batch, y_batch, merged)
                print("train_step:{}, loss:{}, accuracy:{}, recall:{}, precision:{}".format(gp, loss, acc, tpr, pcs))
                train_param["acc"] += acc
                train_param["recall"] += tpr
                train_param["precision"] += pcs
                train_param["ct"] += 1
                writer.add_summary(visual, gp)
                # 验证
                if gp % model.save_every == 0:
                    acc, tpr, pcs, gp, visual = run_eval_step(model, sess, X_vali, Y_vali, merged)
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
        
            # 模型保存1(只保留pb文件)
            save_model_file = os.path.join(model.save_path, 'single_model_bz-%d_e-%d_do-%.2f_l2-%.2f_wz-%d_transvt-%s_pn1vs1-%s_basew2vv5.pb' % (
                                            model.batch_size, model.num_epochs, model.dropout_keep_prob, model.l2_reg_lambda, model.sequence_length, transvt, pn_1vs1))
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Output'])
            with tf.gfile.FastGFile(save_model_file, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            

            with open(os.path.join(save_path, 'recode.txt'), 'a') as f:
                f.write("single_model_bz-%d_e-%d_do-%.2f_l2-%.2f_wz-%d_transvt-%s_%s\n" % (
                                            model.batch_size, model.num_epochs, model.dropout_keep_prob, model.l2_reg_lambda, model.sequence_length, transvt, TIMESTAMP))
                f.write("train:\n acc: %.2f rc: %.2f pcs: %.2f\n" % (train_acc, train_rc, train_pcs))
                f.write("vali:\n acc: %.2f rc: %.2f pcs: %.2f\n" % (vali_acc, vali_rc, vali_pcs))
                
            # reset graph
            tf.reset_default_graph()

            # 执行评测, 未完成
            # os.system('pyhton3 load_model.py %s' % save_model_file)


    # 模型保存2(保存模型文件夹,用于tf_serving的签名保存方法)
    # x = tf.get_default_graph().get_tensor_by_name('Input/Input:0')
    # y = tf.get_default_graph().get_tensor_by_name('predictions:0')
    # builder = tf.saved_model.builder.SavedModelBuilder(model.save_path)
    # signature = tf.saved_model.predict_signature_def(inputs={'input': x}, outputs={'output': y})
    # builder.add_meta_graph_and_variables(sess=sess, tags=['serve'], signature_def_map={'predict': signature})
    # builder.save()




