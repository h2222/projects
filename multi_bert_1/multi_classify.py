import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import jieba
from bert_multitask_learning import (get_or_make_label_encoder, FullTokenizer, 
                                     create_single_problem_generator, train_bert_multitask, 
                                     eval_bert_multitask, DynamicBatchSizeParams, TRAIN, EVAL, PREDICT, BertMultiTask,preprocessing_fn)
import pickle
import types
import os

path = '/home/bairong/shike.shao/multi_bert/bert-multitask-learning'


def load_data(class_name):
    train_file = '/home/bairong/shike.shao/multi_bert/new_data2/'+str(class_name)+'_train.csv'
    dev_file = '/home/bairong/shike.shao/multi_bert/new_data2/'+str(class_name)+'_test.csv'
    train_info = pd.read_csv(train_file, encoding='UTF-8')
    dev_info = pd.read_csv(dev_file, encoding='UTF-8')
    #print('type_combine', list(train_info['type_combine'])[0:1000])

    ## label 映射为数字
    id_label_map = {'invalid':0, 'yes':1, 'no':2, 'deny_money':0}
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

## 初始化params
params = DynamicBatchSizeParams()
params.init_checkpoint = '/home/bairong/shike.shao/chinese_L-12_H-768_A-12'
tf.logging.set_verbosity(tf.logging.DEBUG)
model = BertMultiTask(params)

def cudnngru_hidden(self, features, hidden_feature, mode):
    """
    Hidden of model, will be called between body and top
    Arguments:
    features {dict of tensor} -- feature dict
    hidden_feature {dict of tensor} -- hidden feature dict output by body
    mode {mode} -- ModeKey
    """
    # with shape (batch_size, seq_len, hidden_size)
    seq_hidden_feature = hidden_feature['seq']
    
    cudnn_gru_layer = tf.keras.layers.CuDNNGRU(
            units=self.params.bert_config.hidden_size,
            return_sequences=True,
            return_state=False,
    )
    gru_logit = cudnn_gru_layer(seq_hidden_feature)
    
    return_features = {}
    return_hidden_feature = {}
    
    for problem_dict in self.params.run_problem_list:
        for problem in problem_dict:
            # for slightly faster training
            return_features[problem], return_hidden_feature[problem] = self.get_features_for_problem(
                    features, hidden_feature, problem, mode)
    return return_features, return_hidden_feature


## 将cudnn_hidden函数绑定到对象model上, model可以调用cudnngru_hidden函数
model.hidden = types.MethodType(cudnngru_hidden, model)

# train model
tf.logging.set_verbosity(tf.logging.DEBUG)
pro = 'ask_know|ask_today|ask_tomorrow|identity1|twice|once1|request'
#pro = 'ask_know|ask_today'
train_bert_multitask(problem=pro, num_gpus=1, 
                     num_epochs=10, params=params, 
                     problem_type_dict=new_problem_type, processing_fn_dict=new_problem_process_fn_dict, 
                     model=model, model_dir='model_save')

# evaluate model
print(eval_bert_multitask(problem=pro, num_gpus=1, 
                     params=params, eval_scheme='acc',
                     problem_type_dict=new_problem_type, processing_fn_dict=new_problem_process_fn_dict,
                     model_dir='model_save', model = model))


