import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import jieba
from bert_multitask_learning import (get_or_make_label_encoder, FullTokenizer, 
                                     create_single_problem_generator, train_bert_multitask, 
                                     eval_bert_multitask, predict_bert_multitask, DynamicBatchSizeParams, TRAIN, EVAL, PREDICT, BertMultiTask,preprocessing_fn)
import pickle
import types
import os

path = '/home/bairong/shike.shao/multi_bert/bert-multitask-learning'


def load_data(class_name):
    #train_file = '/home/bairong/shike.shao/multi_bert/data_20000_node/'+str(class_name)+'_train.csv'
    #dev_file = '/home/bairong/shike.shao/multi_bert/data_20000_node/'+str(class_name)+'_test.csv'
    train_file = '/home/bairong/shike.shao/multi_bert/new_data2/'+str(class_name)+'_train.csv'
    dev_file = '/home/bairong/shike.shao/multi_bert/new_data2/'+str(class_name)+'_test.csv'
    train_info = pd.read_csv(train_file, encoding='UTF-8')
    dev_info = pd.read_csv(dev_file, encoding='UTF-8')
    #print('type_combine', list(train_info['type_combine'])[0:1000])

    # label (type_combine)映射 1,2,3,1
    id_label_map = {'invalid':1, 'yes':2, 'no':3, 'deny_money':1}
    train_info['type_combine'] = train_info['type_combine'].apply(lambda x: id_label_map[x])
    dev_info['type_combine'] = dev_info['type_combine'].apply(lambda x: id_label_map[x])
    print(class_name)
    print('lenlen', len(np.array(list(train_info['msg']))))

    return np.array(list(train_info['msg'])), np.array(list(train_info['type_combine'])), np.array(list(dev_info['msg'])), np.array(list(dev_info['type_combine']))

# define new problem
new_problem_type = {'ask_know': 'cls', 'ask_today': 'cls', 'ask_tomorrow': 'cls', 'identity1': 'cls', 'request': 'cls', 'once1': 'cls', 'twice': 'cls'}

@preprocessing_fn
def ask_know(params, mode):
    train_data, train_labels, test_data, test_labels = load_data('ask_know')
    
    if mode == TRAIN:
        input_list = train_data
        target_list = train_labels
    else:
        input_list = test_data
        target_list = test_labels
    
    return input_list, target_list


@preprocessing_fn
def ask_today(params, mode):
    train_data, train_labels, test_data, test_labels = load_data('ask_today')
    
    if mode == TRAIN:
        input_list = train_data
        target_list = train_labels
    else:
        input_list = test_data
        target_list = test_labels
    return input_list, target_list
    

@preprocessing_fn
def ask_tomorrow(params, mode):
    train_data, train_labels, test_data, test_labels = load_data('ask_tomorrow')
    
    if mode == TRAIN:
        input_list = train_data
        target_list = train_labels
    else:
        input_list = test_data
        target_list = test_labels
    return input_list, target_list


@preprocessing_fn
def identity1(params, mode):
    train_data, train_labels, test_data, test_labels = load_data('identity')
    
    if mode == TRAIN:
        input_list = train_data
        target_list = train_labels
    else:
        input_list = test_data
        target_list = test_labels
    return input_list, target_list


@preprocessing_fn
def request(params, mode):
    train_data, train_labels, test_data, test_labels = load_data('request')
    
    if mode == TRAIN:
        input_list = train_data
        target_list = train_labels
    else:
        input_list = test_data
        target_list = test_labels
    return input_list, target_list


@preprocessing_fn
def once1(params, mode):
    train_data, train_labels, test_data, test_labels = load_data('once')
    
    if mode == TRAIN:
        input_list = train_data
        target_list = train_labels
    else:
        input_list = test_data
        target_list = test_labels
    return input_list, target_list


@preprocessing_fn
def twice(params, mode):
    train_data, train_labels, test_data, test_labels = load_data('twice')
    
    if mode == TRAIN:
        input_list = train_data
        target_list = train_labels
    else:
        input_list = test_data
        target_list = test_labels
    return input_list, target_list

new_problem_process_fn_dict = {'ask_know': ask_know, 'ask_today': ask_today, 'ask_tomorrow': ask_tomorrow, 'request': request, 'identity1': identity1, 'once1': once1, 'twice': twice}

# create params and model
params = DynamicBatchSizeParams()
params.init_checkpoint = '/home/bairong/shike.shao/chinese_L-12_H-768_A-12'

def metric(out_true, predict):
    # out_true为 
    # 
    result = {}
    result['out_true'] = out_true
    result['predict'] = predict
    result = pd.DataFrame(result) # [len x 2]

    # 准确度 = TP/(TP+FP+TN+FN)  right = TP+TN,  TP+FP = 所以预测样本
    right = sum(result.predict==result.out_true)
    accuracy = right/result.shape[0]
    
    recal_df = result[result.out_true!=1]
    right = sum(recal_df.predict==recal_df.out_true)
    # 召回率 = TP/(TP+FN)  以关联人举例: 预测问关联人的个数/预测为关联人的个数+预测非关联人为错误的个数(预测非关联人错误为关联人正确)
    recall = right/recal_df.shape[0]
    
    pre_df = result[result.predict!=1]
    right = sum(pre_df.predict==pre_df.out_true)
    precision = right/pre_df.shape[0]
    
    return accuracy, recall, precision


def result(class_name):
    train_x, train_y, pre_x, pre_y = load_data(class_name)
    
    # inputs  训练数据[[训练集用户msg],[训练集label]]   测试数据[[msg],[label]]
    # new_problem_process_fn_dict 例{'identity1':identity1}
    # new_problem_type 例{'identity1':'cls'}

    # 返回预测概率, 
    pred_prob = predict_bert_multitask(inputs=pre_x, 
                                        model_dir='model_save', 
                                        problem=class_name, 
                                        params=params, 
                                        processing_fn_dict=new_problem_process_fn_dict, 
                                        problem_type_dict=new_problem_type)
    predict = []

    # ?
    for prob in pred_prob:
        ner_pred = np.argmax(prob[class_name], axis = -1)
        predict.append(ner_pred)
    
    # evaluation 
    #   out_true 真实标签(pre_y)
    #   predict   预测
    out_true = pre_y
    return out_true, predict

all_out_true, all_predict = [], []
problems = ['ask_know', 'ask_today', 'ask_tomorrow', 'identity1', 'request', 'once1', 'twice']
for problem in problems:
    out_true, predict = result(problem)
    # list扩充 extend 将所以任务的out_true和predict添加入1 list中
    # 为了计算总 precision, recall, accuracy
    all_out_true.extend(out_true)
    all_predict.extend(predict)

    # 打印当前任务
    print('problem', problem)
    # 打印当然任务预测size, (此处size与实际label和实际msg数量相同)
    print('length', len(out_true))
    print('metric', metric(out_true, predict))

print('all_metric', metric(all_out_true, all_predict))
