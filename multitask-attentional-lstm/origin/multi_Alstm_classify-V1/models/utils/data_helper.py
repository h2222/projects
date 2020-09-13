"""
Encoding: utf-8
Author: April
Email: imlpliang@gmail.com
CreateTime: 2019-08-21 16:18
Description: data_helper.py
Version: 1.0
"""
import numpy as np
import pandas as pd
import jieba
import random
from pyltp import Segmentor
import tensorflow as tf
import pickle
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Config.common import seed_num
random.seed(seed_num)




def data_preprocessing_v2(train, test, tokenizer_path, max_len, max_words=50000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    del train_idx, test_idx
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    vocab_size = tokenizer.document_count + 2

    # Save Tokenizer
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Successfully saved the tokenizer to {}".format(tokenizer_path))

    # return train_padded, dev_padded, test_padded, max_words + 2
    return train_padded, test_padded, vocab_size

def build_corpus(seq_info, dict_type2label, config):
    """
    :function: segment_word and de_weight
    :param seq:
    :param concat_in_node:
    :return: [[['1.1', '是', '的'], [0, 1]], [['1.1', '英勇'], [1, 0]], ...]
    """
    # 初始化变量
    min_len_word = config.min_len_word
    concat_in_node = config.concat_in_node
    segment_tool = config.segment
    res_texts = []
    res_labels = []
    res_real_type = []
    seq_l = []

    # 初始化pyltp实例
    segmentor = None
    if segment_tool == 'pyltp':
        print("Using PyLTP to do chinese-word-segment......")
        segmentor = Segmentor()
        segmentor.load(config.cws_model_path)
    else:
        print("Using Jieba to do chinese-word-segment......")

    # 开始遍历
    for item_info in seq_info:
        type_robot, msg, _type = item_info[0], item_info[1], item_info[2]

        temp_text = [type_robot] if concat_in_node and len(type_robot) >= min_len_word else []
        if segment_tool == 'jieba':
            temp_text = temp_text + [word for word in jieba.cut(msg) if len(word) >= min_len_word]
        elif segment_tool == 'pyltp' and segmentor is not None:
            temp_text = temp_text + [word for word in segmentor.segment(msg) if len(word) >= min_len_word]
        text = ' '.join(temp_text)
        text_seq_l = len(temp_text)
        text_seq_len = (text_seq_l if text_seq_l<config.max_len else config.max_len)

        if len(text) != 0:
            res_texts.append(text)
            res_labels.append(dict_type2label[_type])
            seq_l.append(text_seq_len)
            

    res_texts = np.array(res_texts).astype(np.object)
    res_labels = np.array(res_labels).astype(np.int64)
    seq_l = np.array(seq_l).astype(np.int64)
    return res_texts, res_labels, seq_l

# 加载数据的函数
def load_data(config, class_name):
    """
    Returns train set and test set or predict set.
    """
    if config.is_train:
        #根据名字打开文件
        train_file = config.data_dir + '/' + str(class_name)+'_train.csv'
        test_file = config.data_dir + '/' + str(class_name)+'_test.csv'

        # use_cols可选 [type_robot, [msg or msg_del_dup], [type or type_after_merge]]
        curr_target_col = config.use_cols[-1] if config.classification_combine else config.use_cols[-2]
        train_info = pd.read_csv(train_file, encoding='UTF-8', usecols=config.use_cols)
        test_info = pd.read_csv(test_file, encoding='UTF-8', usecols=config.use_cols)

        # 根据实际分类情况，修改label
        train_info, test_info = train_info.values, test_info.values

        print("Train[{}]: {}\nTest[{}]: {}".format(curr_target_col, np.unique(train_info[:, -1]),
                                                                curr_target_col, np.unique(test_info[:, -1])))
        print("Length Train / Test : {} / {}".format(train_info.shape[0], test_info.shape[0]))

        # 建立label与type的相互映射: 下标从0开始
        # type to id
        dict_type2id = np.unique(train_info[:, -1])
        dict_type2id = dict(zip(dict_type2id, range(0, len(dict_type2id))))
        # 固定 invalid_0 yes_1 no_2
        constant = {'invalid': 0, 'yes': 1, 'no': 2}
        for key in constant.keys():
            if key in dict_type2id:
                dict_type2id[key] = constant[key]
        # id to type
        dict_id2type = dict(zip(dict_type2id.values(), dict_type2id.keys()))
        print("Number of classification: {}".format(len(dict_type2id)))
        print("type2id: {}.".format(dict_type2id))
        print("id2type: {}.".format(dict_id2type))

        # 检查 train 是否包含 test 的所有 type 标签
        for item in np.unique(test_info[:, -1]):
            if item not in dict_type2id:
                print("\nIn test set, Train[{}] does not contain {}.".format(curr_target_col, item))


        # shuffle
        for variable in [train_info, test_info]:
            np.random.seed(seed_num)
            np.random.shuffle(variable)

        # build corpus according to format
        x_train, y_train, seq_l_train = build_corpus(train_info, dict_type2id, config)
        x_test, y_test, seq_l_test = build_corpus(test_info, dict_type2id, config)
        class_train = len(x_train)*[class_name]
        class_test = len(x_test)*[class_name]
        return x_train, y_train, class_train, seq_l_train, x_test, y_test, class_test, seq_l_test, dict_id2type
    else:
        # 根据名字打开文件
        predict_file = config.data_dir + '/' + str(class_name) + '_predict.csv'

        curr_target_col = config.use_cols[-1] if config.classification_combine else config.use_cols[-2]

        predict_info = pd.read_csv(predict_file, encoding='UTF-8', usecols=config.use_cols)

        # 根据实际分类情况，修改label
        predict_info= predict_info.values

        print("predict[{}]: {}".format(curr_target_col, np.unique(predict_info[:, -1])))
        print("Length predict : {} ".format(predict_info.shape[0]))

        # 建立label与type的相互映射: 下标从0开始
        # type to id
        dict_type2id = np.unique(predict_info[:, -1])
        dict_type2id = dict(zip(dict_type2id, range(0, len(dict_type2id))))
        # 固定 invalid_0 yes_1 no_2
        constant = {'invalid': 0, 'yes': 1, 'no': 2}
        for key in constant.keys():
            if key in dict_type2id:
                dict_type2id[key] = constant[key]
        # id to type
        dict_id2type = dict(zip(dict_type2id.values(), dict_type2id.keys()))
        print("Number of classification: {}".format(len(dict_type2id)))
        print("type2id: {}.".format(dict_type2id))
        print("id2type: {}.".format(dict_id2type))

        # shuffle
        for variable in [predict_info]:
            np.random.seed(seed_num)
            np.random.shuffle(variable)

        # build corpus according to format
        x_predict, y_predict, seq_l_predict = build_corpus(predict_info, dict_type2id, config)
        class_predict = len(x_predict)*[class_name]

        return  x_predict, y_predict, class_predict, seq_l_predict, dict_id2type


def load_all_data(config):
    if config.is_train:
        x_train_a = []
        y_train_a = []
        class_train_a = []
        x_test_a = []
        y_test_a = []
        class_test_a = []
        real_type_train_a = []
        real_type_test_a = []
        seq_l_train_a=[]
        seq_l_test_a =[]
        node_name = [str(i) for i in config.cls_type]#获取类别名称

        for name in node_name:
            print('name', name)

            x_train, y_train, class_train, seq_l_train, x_test, y_test, class_test, seq_l_test, dict_id2type = load_data(config, name)
            print('dict_id', dict_id2type)
            x_train_a.extend(x_train)
            y_train_a.extend(y_train)
            class_train_a.extend(class_train)
            seq_l_train_a.extend(seq_l_train)
            x_test_a.extend(x_test)
            y_test_a.extend(y_test)
            class_test_a.extend(class_test)
            seq_l_test_a.extend(seq_l_test)

            

        return [x_train_a, y_train_a, class_train_a, seq_l_train_a, x_test_a, y_test_a, class_test_a, seq_l_test_a], dict_id2type
    else:
        x_predict_a = []
        y_predict_a = []
        class_predict_a = []
        real_type_predict_a = []
        seq_l_predict_a =[]
        node_name = [str(i) for i in config.cls_type]  # 获取类别名称

        for name in node_name:
            print('name', name)
            x_predict, y_predict, class_predict, seq_l_predict, dict_id2type = load_data(config, name)
            print('dict_id', dict_id2type)
            x_predict_a.extend(x_predict)
            y_predict_a.extend(y_predict)
            class_predict_a.extend(class_predict)
            seq_l_predict_a.extend(seq_l_predict)

        return [x_predict_a, y_predict_a, class_predict_a, seq_l_predict_a], dict_id2type

def fill_feed_dict(data_X, data_Y, class_train, seq_l, data_Text_Real, batch_size):
    """Generator to yield batches"""

    x_batch = []
    y_batch = []
    text_real_batch = []
    class_name_batch = []
    seq_l_batch = []
    for idx in range(data_X.shape[0] // batch_size):
        class_train1 = class_train[batch_size * idx: batch_size * (idx + 1)]
        if len(set(class_train1)) == 1:
            x_batch.append(data_X[batch_size * idx: batch_size * (idx + 1)])
            y_batch.append(data_Y[batch_size * idx: batch_size * (idx + 1)])
            text_real_batch.append(data_Text_Real[batch_size * idx: batch_size * (idx + 1)])
            class_name_batch.append(class_train1[0])
            seq_l_batch.append(seq_l[batch_size * idx: batch_size * (idx + 1)])
    shuffled_X, shuffled_Y, shuffle_class, shuffle_seq_l, shuffle_Text_Real = shuffle(x_batch, y_batch, class_name_batch, seq_l_batch, text_real_batch)
    print('len', len(shuffled_X))
    for x, y, class_, seq_len, text_real in zip(shuffled_X, shuffled_Y, shuffle_class, shuffle_seq_l, shuffle_Text_Real):
        yield x, y, class_, seq_len, text_real

def save_forecast_sample(prediction_labels, target_labels, texts_reals, dict_id2type, save_csv_path, classify_name, epoch):
    # 预测的句子 texts
    # 真正的标签 real_types
    texts = texts_reals
    # 预测的标签和概率 prediction_label
    # 正确的标签 target_labels
    prediction_labels = [dict_id2type[x] for x in prediction_labels]
    target_labels = [dict_id2type[x] for x in target_labels]
    prediction_labels = np.array(prediction_labels)
    target_labels = np.array(target_labels)
    classify_name = np.array(classify_name)
    # 是否正确 is_correct
    is_correct = (prediction_labels == target_labels)

    # write to csv
    columns_name = ["is_correct", "predict", "target",  "texts", "classify"]
    col_value = [is_correct, prediction_labels, target_labels,  texts, classify_name]
    forecast_sample = pd.DataFrame(dict(zip(columns_name, col_value)), columns=columns_name)
    # Write to multiple files
    save_csv_path = save_csv_path + "-epoch{}.csv".format(str(epoch))
    forecast_sample.to_csv(save_csv_path, index=True, index_label="id", encoding='utf-8-sig', mode='w')

def my_evaluation(predictions, targets, loss, dict_id2type):
    dict_precision = dict(zip(dict_id2type.values(), [0 for _ in range(len(dict_id2type))]))
    dict_recall = dict(zip(dict_id2type.values(), [0 for _ in range(len(dict_id2type))]))
    predictions = np.array(predictions)
    targets = np.array(targets)
    # accuracy
    numerator = (predictions == targets)
    denominator = len(targets)
    accuracy = numerator.sum() / denominator

    # precision - multiple classification
    for curr_label, curr_type in dict_id2type.items():
        numerator = ((predictions == targets) & (targets == curr_label))
        denominator = (predictions == curr_label)
        # print(numerator.sum(), denominator.sum())
        if denominator.sum() != 0:
            dict_precision[curr_type] = numerator.sum() / denominator.sum()
    # recall - multiple classification
    for curr_label, curr_type in dict_id2type.items():
        numerator = ((predictions == targets) & (targets == curr_label))
        denominator = (targets == curr_label)
        # print(numerator.sum(), denominator.sum())
        if denominator.sum() != 0:
            dict_recall[curr_type] = numerator.sum() / denominator.sum()

    # Macro-Averaging
    macro_precision_sum = sum([x for x in dict_precision.values()])
    macro_precision = macro_precision_sum / len(dict_precision)
    macro_recall_sum = sum([x for x in dict_recall.values()])
    macro_recall = macro_recall_sum / len(dict_recall)
    # Macro-Averaging f_measure 第一种计算方式
    macro_f_measure = macro_precision * macro_recall * 2 / (macro_precision + macro_recall)
    # Micro-Averaging
    dict_tp, dict_fp, dict_fn = {}, {}, {}
    for curr_label in dict_id2type.keys():
        tp_bool = ((predictions == targets) & (targets == curr_label))
        fp_bool = ((predictions == curr_label) & (targets != curr_label))
        fn_bool = ((predictions != targets) & (targets == curr_label))
        dict_tp[curr_label] = tp_bool.sum()
        dict_fp[curr_label] = fp_bool.sum()
        dict_fn[curr_label] = fn_bool.sum()
    sum_tp4all_label = sum(list(dict_tp.values()))
    sum_fp4all_label = sum(list(dict_fp.values()))
    sum_fn4all_label = sum(list(dict_fn.values()))
    micro_precision = sum_tp4all_label / (sum_tp4all_label + sum_fp4all_label)
    micro_recall = sum_tp4all_label / (sum_tp4all_label + sum_fn4all_label)

    # 打印信息
    print("Loss {:.5f}, Accuracy {:.5f}".format(loss, accuracy))
    print("Every Precision: ", end='')
    for key, val in dict_precision.items():
        print("{}: {:.5f}, ".format(key, val), end='')
    print()
    print("Every Recall:    ", end='')
    for key, val in dict_recall.items():
        print("{}: {:.5f}, ".format(key, val), end='')
    print()
    return accuracy


