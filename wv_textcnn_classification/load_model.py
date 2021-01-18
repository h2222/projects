import tensorflow as tf
import os
import sys
sys.path.append('test')
import numpy as np
import re
from sklearn.metrics import auc
from sklearn import metrics 
from googletrans import Translator
from request import request_single_simple, request_single_simple_local, content_d
import pandas as pd
import copy
import joblib

from gensim.models import KeyedVectors

# sentence length
sentence_len = 0

def load_textcnn_model_file(pb_path):
    sess = tf.Session()
    with tf.gfile.GFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    return sess


def model_predict(sess, X_test=None, Y_test=None):
    sess.run(tf.global_variables_initializer())
    input_vec = sess.graph.get_tensor_by_name("Input_x:0")
    # input_y = sess.graph.get_tensor_by_name("Input_y:0")
    prediction = sess.graph.get_tensor_by_name("Output:0")
    if np.all(X_test == None) or np.all(Y_test == None):
        print("use default test set")
        X_test = np.load('test/X_test_with_atten.npy')
        Y_test = np.load('test/Y_test_with_atten.npy')
    feed_dict = {input_vec:X_test}
    prob = sess.run(prediction, feed_dict)
    # tf.reset_default_graph()
    return prob, Y_test


def load_RF_PCA_model_file(pac_model, rf_model, X_test, Y_test):
    # X_test_reduce = pac_model.transform(X_test)
    Y_pred = rf_model.predict(X_test)
    return Y_pred


def single_simple_pred(pb_path, sentence_len, w2v_path=None, use_local_w2v=True):
    trans = Translator()
    sess = load_textcnn_model_file(pb_path)
    w2v_model = None
    if w2v_path != None:
        w2v_model = KeyedVectors.load_word2vec_format(w2v_path)
    
    # test no_trip case 60
    test_no_trip_60_case_f = open('test/total_case_v3_no_change.csv', 'r')
    test_no_trip_60_case_f.readline()
  
    # test_no_trip_200_case
    test_no_trip_200_case_f = open('test/total_case_v4_no_change.csv', 'r')
    test_no_trip_200_case_f.readline()

    # test_no_trip_57352 10月份数据
    test_no_trip_month_10_f = open('test/data_4_5_9_10.csv_no_trip.csv', 'r')
    test_no_trip_month_10_f = test_no_trip_month_10_f.readlines()[23751:]


    # for statistic    
    bad_case_path = 'test/bad_case_v4.csv'
    good_case_path = 'test/good_case_v4.csv'
    unknow_case_path = 'test/unknow_case_v4.csv'
    open(bad_case_path, 'w')
    open(good_case_path, 'w')
    open(unknow_case_path,'w')

    for i, line in enumerate(test_no_trip_200_case_f.readlines()):
        if i == -1:
            break
        # semantic_label = "乘客未上车"
        semantic_label = line.replace('\n', '').split('|')[-1]
        Y_test = np.array([[1, 0]])
        content_d['seq_len'] = sentence_len

        if use_local_w2v:
            wv, wv_mean, ticket_id, content = request_single_simple_local(line, w2v_model, sentence_len, test=None)
            if isinstance(wv, str):
                print("wv not exist")
                continue
            wv_mean = wv_mean.reshape(1, -1)
        else:
            wv, wv_mean, ticket_id, content = request_single_simple(line)
            wv = wv.reshape(1, -1)
            wv_mean = wv_mean.reshape(1, -1)
        
        prob, y = model_predict(sess, X_test=wv, Y_test=Y_test)
        pred = np.argmax(prob, 1)
        target = np.argmax(Y_test, 1)
        
        if prob[0, 1] > 0.5: #0.87:
            print('bad case')
            with open(bad_case_path, 'a') as f:
                # try:
                #     trans_text = trans.translate(content, dest='zh-CN', src='pt').text
                # except:
                trans_text = "trans"
                bad_case = ticket_id + "|" + content + "|" + str(prob[-1,-1]) + "|" + str(pred[-1]) + "|" + trans_text + "|" + semantic_label + "\n"
                print(bad_case)
                f.write(bad_case)
        else:
            print('good case')
            with open(good_case_path, 'a') as f:
                # try:
                #     trans_text = trans.translate(content, dest='zh-CN', src='pt').text
                # except:
                trans_text = "trans"
                good_case = ticket_id + "|" + content + "|" + str(prob[-1, -1]) + "|" + str(pred[-1]) + "|" + trans_text + "|" + semantic_label + "\n"
                print(good_case)
                f.write(good_case)




def evaluation(prob, Y_test):
    sess2 = tf.Session()
    init = tf.initialize_all_variables()
    sess2.run(init)
    y_pred = tf.argmax(prob, 1, name="predictions")
    actuals = tf.argmax(Y_test, 1)
    correct_predictions = tf.equal(y_pred, actuals)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(y_pred)
    zeros_like_predictions = tf.zeros_like(y_pred)
    # ture possitive rate
    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(y_pred, zeros_like_predictions)
            ),
            "float"))

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(y_pred, ones_like_predictions)
            ),
            "float"))

    # false negative rate
    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), # [true, false, ...] 0的地方为false, 1的地方为true, 表示原始预测集为正样本的地方为true
                tf.equal(y_pred, ones_like_predictions) # [true, true, false, false, true,..] 0的地方就是true, 1的地方是false, 表示模型预测为负样本的地方true
            ),
            "float")) # 计算有多少 true-true 
    # false possitive rate
    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(y_pred, zeros_like_predictions)
            ),
            "float"))
    
    # acc, recall, precision
    accuracy, tp, tn, fp, fn = sess2.run([accuracy, tp_op, tn_op, fp_op, fn_op])
    recall = float(tp) / (float(tp) + float(fn) + 1e-7)     # 预测正确为1 / 真是样本的1
    precision = float(tp) / (float(tp) + float(fp) + 1e-7)  # 预测正确为1 / 模型给预测的1
    
    # roc, auc
    Y_test_1d = np.argmax(Y_test, 1)
    prob_1d = np.argmax(prob, 1)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test_1d, prob_1d)
    auc = metrics.auc(fpr, tpr)

    print("tp:{:g}, tn:{:g} fp:{:g}, fn:{:g}".format(float(tp), float(tn), float(fp), float(fn)))
    print("accuracy:{:g}, recall:{:g}, precision:{:g}".format(accuracy, recall, precision))
    print("auc : {}".format(auc))


def translate(path):
    trans = Translator()
    trans_file_name = path.split('.')[0] + '__trans.csv'
    with open(path, 'r') as f:
        f.readline()
        for i, line in enumerate(f.readlines()[9840:]):
            print(i)
            try:
                line_new = line.replace('\n', '') + "|" + trans.translate(line.split('|')[2], dest='zh-CN', src='pt').text + "\n"
                with open(trans_file_name, 'a', encoding='utf-8') as f2:
                    f2.write(line_new)
            except Exception as e:
                print(e)
                continue


def find_best_threshold(path, thresholds):
    df = pd.read_csv(path, sep='|', encoding='utf-8-sig')
    print(df.shape)
    total_test_case = 199
    for thresh in thresholds:
        # print(df.shape)
        df_temp = df.loc[df['prob'] > thresh]
        total = len(df_temp)
        num_no_trip = len(df_temp.loc[df_temp['label'] == '乘客未上车'])
        num_trip = len(df_temp.loc[df_temp['label'] == '乘客上车'])
        
        recall_rate = num_trip / (total_test_case + 1e-10)
        recall_acc_rate = num_trip / (total + 1e-10)

        print("threshold:{}, 模型预测为上车的数量:{}, 卡阈值后模型预测为上车的数量:{}, 真正未上车的数量:{}, 真正上车的数量:{}, 上车召回率:{}, 上车召回准确率:{}".format(thresh, 48, total, num_no_trip, num_trip, recall_rate, recall_acc_rate))

    


if __name__ == "__main__":
    path1 = sys.argv[1]
    # path2 = sys.argv[2]
    # print(path1, path2)
    sentence_len = 30

    # trans
    # translate(path1)

    # single test
    # single_simple_pred(path1, sentence_len, w2v_path=path2)
    
    # total
    # sess = load_textcnn_model_file(path1)
    # prob, Y_test = model_predict(sess)
    # evaluation(prob, Y_test)

    # best threshold test
    thresholds = [ round(0.5 + 0.01 * i, 2) for i in range(20)]
    print(thresholds)
    find_best_threshold(path1, thresholds)



    ###  单步测试
    # line = "360287970317045800|Motorista pegou o passageiro errado e cobrou do meu cartão|0|0|司机接错了乘客并向我收费|[[0.57934713 0.42065284]]|乘客未上车"
    # # v,_,_,_ = request_single_simple(line=line)  
    # # print(v.tolist())
    # w2v_model = KeyedVectors.load_word2vec_format(path2)
    # from nltk.corpus import stopwords
    # from nltk import tokenize
    # target = "Motorista parou na minha porta e foi embora"
    # sw = stopwords.words('portuguese')
    # target_ls = target.split(' ')
    # print('space split: ', target_ls)
    # target_ls = tokenize.word_tokenize(target, language='portuguese')
    # print('tokenizer:', target_ls)
    # res = []
    # for s in target_ls:
    #     if s not in sw:
    #         res.append(s)
    # target = ' '.join(res)
    # target += '|001'
    # print(target)
    # wv, _, _, _ = request_single_simple_local(line, w2v_model, 30, test=target)
    # Y_test = np.array([[1, 0]])
    # sess = load_textcnn_model_file(path1)
    # prob, y = model_predict(sess, X_test=wv, Y_test=Y_test)
    # print(prob)



    