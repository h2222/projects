"""
Encoding: utf-8
Author: April
Email: imlpliang@gmail.com
CreateTime: 2019-08-21 16:18
Description: data_helper.py
Version: 1.0
"""
import numpy as np
import re
import datetime
import pandas as pd
import jieba
import random
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
#random.seed(seed_num)


# 过滤函数
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def del_duplicate_in_diff_set(info, no_contain_info, info_name):
    """
    # 删除重复的文本 在两个不同的数据集下
    :param train:
    :param dev_test:
    :param curr_name:
    :return:
    """
    set_only = set(list(no_contain_info[:, 1]))
    new_info = []
    for item in info:
        text = item[1]
        if text not in set_only:
            set_only.add(text)
            new_info.append(item)
    print("Delete duplicate corpus in {}, according to other sets, the number of duplicates deleted: {}."
          .format(info_name, len(info)-len(new_info)))
    return np.array(new_info)


def fix_label(train_info, dev_info, test_info, label_map, filter_invalid):
    if filter_invalid:
        train_info = [x for x in train_info if x[2] != '无效语义']
        dev_info = [x for x in dev_info if x[2] != '无效语义']
        test_info = [x for x in test_info if x[2] != '无效语义']
    for item in train_info:
        item[3] = label_map[item[2]]
    for item in dev_info:
        item[3] = label_map[item[2]]
    for item in test_info:
        item[3] = label_map[item[2]]
    return np.array(train_info), np.array(dev_info), np.array(test_info)


def del_duplicate(info, info_name):
    """
    # 删除重复的文本 在同一个数据集下
    :param info:
    :param info_name:
    :return:
    """
    set_only = set()
    new_info = []
    for item in info:
        text = item[1]
        if text not in set_only:
            set_only.add(text)
            new_info.append(item)
    print("Delete duplicate corpus in {}, the number of duplicates deleted: {}."
          .format(info_name, len(info)-len(new_info)))
    return np.array(new_info)


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
    add_pinyin = config.add_pinyin
    segment_tool = config.segment
    res_texts = []
    res_labels = []
    res_real_type = []

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
        type_robot, msg, real_type, _type = item_info[0], item_info[1], item_info[2], item_info[3]

        # 分词: 两种情况（拼接 无拼接） 如果拼接in_node, 则提前加入temp_add中
        # temp_text = [str(type_robot).replace('.', '-')] if concat_in_node and min_len_word == 0 else []
        temp_text = [type_robot] if concat_in_node and len(type_robot) >= min_len_word else []
        if segment_tool == 'jieba':
            temp_text = temp_text + [word for word in jieba.cut(msg) if len(word) >= min_len_word]
        elif segment_tool == 'pyltp' and segmentor is not None:
            temp_text = temp_text + [word for word in segmentor.segment(msg) if len(word) >= min_len_word]
        text = ' '.join(temp_text)

        # 是否加入拼音
        if add_pinyin == 'pinyin':
            pinyin_str = ' '.join([x[0] for x in pinyin(item_info[1])])
            text = text + ' ' + pinyin_str
        elif add_pinyin == 'lazy_pinyin':
            pinyin_str = ' '.join([x for x in lazy_pinyin(item_info[1])])
            text = text + ' ' + pinyin_str

        # TODO: 停用词

        if len(text) != 0:
            res_texts.append(text)
            res_labels.append(dict_type2label[_type])
            res_real_type.append(real_type)

    res_texts = np.array(res_texts).astype(np.object)
    res_labels = np.array(res_labels).astype(np.int64)
    res_real_type = np.array(res_real_type)
    return res_texts, res_labels, res_real_type


# 加载数据的函数
def load_data(config, class_name, label_map):
    """
    Returns train set and test set.
    """
    train_file = '/home/shike.shao/new_lstm_attention/multi_Alstm_classify/data_test/'+'small_'+str(class_name)+'_class3_train.csv'
    test_file = '/home/shike.shao/new_lstm_attention/multi_Alstm_classify/data_test/'+'small_'+str(class_name)+'_class3_test.csv'
    dev_file = '/home/shike.shao/new_lstm_attention/multi_Alstm_classify/data_test/'+'small_'+str(class_name)+'_class3_dev.csv'
    # use_cols可选 [type_robot, [msg or msg_del_dup], [type or type_after_merge]]
    curr_target_col = config.use_cols[-1] if config.classification_combine else config.use_cols[-2]
    train_info = pd.read_csv(train_file, encoding='UTF-8', usecols=config.use_cols)
    dev_info = pd.read_csv(dev_file, encoding='UTF-8', usecols=config.use_cols)
    test_info = pd.read_csv(test_file, encoding='UTF-8', usecols=config.use_cols)

    # 根据实际分类情况，修改label
    train_info, dev_info, test_info = train_info.values, dev_info.values, test_info.values
    train_info, dev_info, test_info = fix_label(train_info, dev_info, test_info, label_map, config.filter_invalid)

    print("Train[{}]: {}\nDev[{}]: {}\nTest[{}]: {}".format(curr_target_col, np.unique(train_info[:, -1]),
                                                            curr_target_col, np.unique(dev_info[:, -1]),
                                                            curr_target_col, np.unique(test_info[:, -1])))
    print("Length Train / Dev / Test : {} / {} / {}".format(train_info.shape[0], dev_info.shape[0], test_info.shape[0]))

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

    # 检查 train 是否包含 dev test 的所有 type 标签
    for item in np.unique(dev_info[:, -1]):
        if item not in dict_type2id:
            print("\nIn dev set, Train[{}}] does not contain {}.".format(curr_target_col, item))
    for item in np.unique(test_info[:, -1]):
        if item not in dict_type2id:
            print("\nIn test set, Train[{}] does not contain {}.".format(curr_target_col, item))

    # 查重去重: 三个数据集中数据不可重复
    print("\nCheck if text is duplicated")
    #train_info = del_duplicate(train_info, "train")
    #dev_info = del_duplicate(dev_info, "dev")
    #test_info = del_duplicate(test_info, "test")
    train_info = del_duplicate_in_diff_set(train_info, np.concatenate((dev_info, test_info), axis=0), "train")
    dev_info = del_duplicate_in_diff_set(dev_info, test_info, "dev")

    # shuffle
    for variable in [train_info, dev_info, test_info]:
        np.random.seed(seed_num)
        np.random.shuffle(variable)

    # build corpus according to format
    x_train, y_train, real_type_train = build_corpus(train_info, dict_type2id, config)
    x_dev, y_dev, real_type_dev = build_corpus(dev_info, dict_type2id, config)
    x_test, y_test, real_type_test = build_corpus(test_info, dict_type2id, config)
    class_train = len(x_train)*[class_name] 
    class_dev = len(x_dev)*[class_name] 
    class_test = len(x_test)*[class_name] 
    return x_train, y_train, class_train, x_dev, y_dev, class_dev, x_test, y_test, class_test, real_type_train, real_type_dev, real_type_test, dict_id2type

def fix_label2(train_info, dev_info, label_map):
    #train_info = [x for x in train_info if x[2] != '无效语义']
    #dev_info = [x for x in dev_info if x[2] != '无效语义']
    #test_info = [x for x in test_info if x[2] != '无效语义']
    for item in train_info:
        item[1] = label_map[item[1]]
    for item in dev_info:
        item[1] = label_map[item[1]]
    for item in test_info:
        item[1] = label_map[item[1]]
    return np.array(train_info), np.array(dev_info), np.array(test_info)

def load_data2(class_name, label_map):
    train_file = '/home/shike.shao/bert-multitask-learning2/bert-multitask-learning/Data/'+'small_'+str(class_name)+'_class3_train.csv'
    dev_file = '/home/shike.shao/bert-multitask-learning2/bert-multitask-learning/Data/'+'small_'+str(class_name)+'_class3_dev.csv'
    train_info = pd.read_csv(train_file, encoding='UTF-8')
    dev_info = pd.read_csv(dev_file, encoding='UTF-8') 
    train_
    # 根据实际分类情况，修改label
    train_info, dev_info = train_info.values, dev_info.values
    #train_info, dev_info = fix_label2(train_info, dev_info, label_map)
    print(dev_info)
    print(dev_info[:1])
    return train_info[0], train_info[1], dev_info[0], dev_info[1]

def load_all_data():
    x_train_a = []
    y_train_a = []
    class_train_a = []
    x_dev_a = []
    y_dev_a = []
    class_dev_a = []
    x_test_a = [] 
    y_test_a = []
    class_test_a = []
    real_type_train_a = []
    real_type_dev_a = []
    real_type_test_a = []
    flag = 0
    if flag == 0:
        label_map_ask_know = {'无效语义':'invalid', '肯定': 'yes', '关联人': 'yes', '否定': 'no', '非关联人': 'no'}
        label_map_ask_today = {'无效语义':'invalid', '肯定': 'yes', '同意今日还款': 'yes', '否定': 'no', '不同意今日还款': 'no', '同意明日还款': 'no', '不同意明日还款': 'no'}
        label_map_ask_tomorrow = {'无效语义':'invalid', '肯定': 'yes', '同意明日还款': 'yes', '同意今日还款': 'yes', '否定': 'no', '不同意明日还款': 'no'}
        label_map_identity = {'无效语义':'invalid', '肯定': 'yes', '本人': 'yes', '否定': 'no', '非本人': 'no', '关联人': 'no', '非关联人': 'no'}
        label_map_once = {'无效语义':'invalid', '不认可金额': 'invalid', '肯定': 'yes', '清楚借款': 'yes', '同意今日还款': 'yes', '不同意今日还款': 'yes', '同意明日还款': 'yes', '不同意明日还款': 'yes', '否定': 'no', '不清楚借款': 'no', '遗忘语境': 'no'}
        label_map_request = {'无效语义':'invalid', '肯定': 'yes','同意转告': 'yes','否定': 'no','不同意转告': 'no','非关联人': 'no'}
        label_map_twice = {'无效语义':'invalid', '肯定': 'yes', '清楚借款': 'yes', '回忆起借款': 'yes', '同意今日还款': 'yes', '不同意今日还款': 'yes', '同意明日还款': 'yes', '不同意明日还款': 'yes', '否定': 'no', '不清楚借款': 'no', '遗忘语境': 'no'} 
    else:
        label_map_ask_know = {'无效语义':'invalid', '肯定': 'valid', '关联人': 'valid', '否定': 'valid', '非关联人': 'valid'}
        label_map_ask_today = {'无效语义':'invalid', '肯定': 'valid', '同意今日还款': 'valid', '否定': 'valid', '不同意今日还款': 'valid', '同意明日还款': 'valid', '不同意明日还款': 'valid'}
        label_map_ask_tomorrow = {'无效语义':'invalid', '肯定': 'valid', '同意明日还款': 'valid', '同意今日还款': 'valid', '否定': 'valid', '不同意明日还款': 'valid'}
        label_map_identity = {'无效语义':'invalid', '肯定': 'valid', '本人': 'valid', '否定': 'valid', '非本人': 'valid', '关联人': 'valid', '非关联人': 'valid'}
        label_map_once = {'无效语义':'invalid', '不认可金额': 'invalid', '肯定': 'valid', '清楚借款': 'valid', '同意今日还款': 'valid', '不同意今日还款': 'valid', '同意明日还款': 'valid', '不同意明日还款': 'valid', '否定': 'valid', '不清楚借款': 'valid', '遗忘语境': 'valid'}
        label_map_request = {'无效语义':'invalid', '肯定': 'valid','同意转告': 'valid','否定': 'valid','不同意转告': 'valid','非关联人': 'valid'}
        label_map_twice = {'无效语义':'invalid', '肯定': 'valid', '清楚借款': 'valid', '回忆起借款': 'valid', '同意今日还款': 'valid', '不同意今日还款': 'valid', '同意明日还款': 'valid', '不同意明日还款': 'valid', '否定': 'valid', '不清楚借款': 'valid', '遗忘语境': 'valid'} 
    node_name = ['ask_know', 'ask_today', 'ask_tomorrow', 'identity', 'once', 'request', 'twice']
    label_maps = [label_map_ask_know, label_map_ask_today, label_map_ask_tomorrow, label_map_identity, label_map_once, label_map_request, label_map_twice]
    #node_name = ['ask_know']
    #label_maps = [label_map_ask_know]
    for name, label_map in zip(node_name, label_maps):
        print('naem', name)
        print('map', label_map)
        x_train, y_train, x_dev, y_dev = load_data2(name, label_map)
        x_train_a.extend(x_train)
        y_train_a.extend(y_train)
        x_dev_a.extend(x_dev)
        y_dev_a.extend(y_dev)

    return x_train_a, y_train_a, x_dev_a, y_dev_a   

def save_forecast_sample(prediction_labels, target_labels, texts_reals, dict_id2type, save_csv_path, classify_name, epoch):
    # 预测的句子 texts
    # 真正的标签 real_types
    texts, real_types = zip(*texts_reals)
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
    columns_name = ["is_correct", "predict", "target", "real_target", "texts", "classify"]
    col_value = [is_correct, prediction_labels, target_labels, real_types, texts, classify_name]
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
    """
    # Macro-Averaging f_measure 第二种计算方式
    dict_f_measure = {}
    for key in dict_label2type.keys():
        p_val = dict_precision[key]
        r_val = dict_recall[key]
        if p_val + r_val != 0:
            dict_f_measure[key] = p_val * r_val * 2 / (p_val + r_val)
    macro_f_measure_second = sum([x for x in dict_f_measure.values()]) / len(dict_f_measure)
    """
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
    micro_f_measure = micro_precision * micro_recall * 2 / (micro_precision + micro_recall)

    # 打印信息
    print("Loss {:.5f}, Accuracy {:.5f}".format(loss, accuracy))
    #print("Macro-Averaging: Precision {:.5f}, Recall {:.5f}, F-measure {:.5f}"
    #      .format(loss, accuracy, macro_precision, macro_recall, macro_f_measure))
    #print("Micro-Averaging: Precision {:.5f}, Recall {:.5f}, F-measure {:.5f}"
    #      .format(loss, accuracy, micro_precision, micro_recall, micro_f_measure))
    print("Every Precision: ", end='')
    for key, val in dict_precision.items():
        print("{}: {:.5f}, ".format(key, val), end='')
    print()
    print("Every Recall:    ", end='')
    for key, val in dict_recall.items():
        print("{}: {:.5f}, ".format(key, val), end='')
    print()
    # return accuracy, macro_f_measure, micro_f_measure
    #return macro_f_measure
    return accuracy

if __name__ == '__main__':
    load_all_data()
