# coding=utf-8

import pandas as pd
import re
import os
from random import randint
from sklearn.utils import shuffle


def stop_word(sw_path):
    with open(sw_path, 'r', encoding='utf_8') as f:
        swl = []
        for i in f.readlines():
            swl.append(i.strip('\n'))
        return swl


def load_process(path, swl):
    
    if 'train' in path:
        df_train = pd.read_csv(path, encoding='utf-8')
        df_train['comment'] = df_train['comment'].map(lambda x : ''.join(re.findall('[0-9A-Za-z\s\t]', str(x))))

    # get label
    train_label = list(df_train['Expected'])

    # rough selection
    rough_train_list = []
    for i in df_train['comment']:
        rough_train_list.append(i.split(' '))

    # detail selection
    train_list = []    
    for line in rough_train_list:
        newl = [i for i in line if len(i) <= 10 and i !='' and not i in swl]
        train_list.append(' '.join(newl))


    # return train_list, 
    print('train x length:', len(train_list))
    print('train y length', len(train_label))
    # 释放资源
    del rough_train_list
    del swl
    return train_list, train_label
        


def get_data():
    path = '../dataset/train.csv'
    sw_path = './utils/tool/stop_w.txt'
    swl = stop_word(sw_path)
    x, y = load_process(path, swl=swl)

    return x, y




def build_vocab(path):
    if os.path.exists(path):
        os.remove(path)
        open(path, 'w', encoding='utf-8')

    x, y = get_data()
    
    vocab_dict = {}
    for line in x:
        for w in line.split(' '):
            if not w in vocab_dict:
                vocab_dict[w] = 0
                vocab_dict[w] += 1
            else:
                vocab_dict[w] += 1
    
    order_vocab = sorted(vocab_dict.items(), key=lambda  x: x[1], reverse=True)
    print(order_vocab)

    with open(path, 'a', encoding='utf-8') as f:
        for w, t in order_vocab:
            f.write(w+'\r')



def fill_feed_dict(data_x,  data_y, class_name, batch_size):

    batch_x = []
    batch_y = []
    batch_seq_l = []
    batch_cname = []

    # 补全尾部batch
    remain = len(data_x) % batch_size
    bindex = [(s, s+batch_size-1) for s in range(0, len(data_x), batch_size)]
    bindex[-1] = (bindex[-2][1]+1, bindex[-2][1]+remain)
    
    print(bindex)
    # 数据分段
    for s, e in bindex:
        batch_x += [data_x[s:e]]
        batch_y += [data_y[s:e]]
        batch_cname += [class_name]
        batch_seq_l += [len(i.split(' ')) for i in data_x[s:e]]

    batch_xs, batch_ys, batch_csname, batch_seq_ls = shuffle(batch_x, batch_y, batch_cname, batch_seq_l)

    # 迭代数据
    # 返回类型, x list of segemeted sentence,  y list of int 0 or 1, name list of class name, length == x and y
    for x, y, name in zip(batch_xs, batch_ys, batch_csname, batch_seq_ls):
        yield x, y, name, l


if __name__ == "__main__":
    pass
    # build_vocab('./tool/vocab.txt')
    # x, y = get_data()
    # print(x)
    # print(y)