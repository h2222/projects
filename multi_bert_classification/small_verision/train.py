#coding=utf-8

# import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import types
import os

from params import Param




def preprocessing_fn(func):
    def wapper(param, mode, task_name):
        print(func.__name__)
        print(param.tmp_dir)        
        pickle_file  = os.path.join(param.tmp_dir, "%s_%s_data.pkl" % (task_name, mode))
        if param.task_type[task_name] == param.cls:
            
            print('分类任务文本处理')


    return wapper









@preprocessing_fn
def load_data(param, mode, task_name):
    train_file = task_name+'_train.csv'
    dev_file = task_name+'_test.csv'
    train_df = pd.read_csv(train_file, sep='|', encoding='UTF-8')
    dev_df = pd.read_csv(dev_file, sep='|', encoding='UTF-8')
    # label 映射为数字
    id_label_map = {'invalid':0, 'yes':1, 'no':2, 'deny_money':0}
    train_df['label'] = train_df['label'].apply(lambda x: id_label_map[x])
    dev_df['label'] = dev_df['label'].apply(lambda x: id_label_map[x])
    # 任务信息
    print('任务名称: %s, 任务数据量:%d' % (task_name, len(train_df)))

    if mode == 'train':
        return np.array(list(train_df['msg'])), np.array(list(train_df['label']))
    elif mode == 'eval':
        return np.array(list(dev_df['msg'])), np.array(list(dev_df['label']))





if __name__ == "__main__":
    param = Param()
    load_data(param, 'train', 'task1')

    # print(train_x)
    # print(train_y)
    # print(test_x)
    # print(test_y)









