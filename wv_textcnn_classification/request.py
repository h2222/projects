# coding=utf-8
import sys, os
import re
import json

import numpy as np
import requests
import pandas as pd

from nltk.corpus import stopwords
from nltk import tokenize
from gensim.models import KeyedVectors 
import datetime
from googletrans import Translator


# word2vec interface params
url = "http://10.186.1.232:8081/api/v2/pt_emb"
header = "content-type: application/json"
content_d = {"text": "default", "seq_len": 30}
sw = stopwords.words('portuguese')
trans = Translator()


def preprocess_data(content):
    word_ls =  tokenize.word_tokenize(content, language='portuguese') 
    word_ls = [i for i in word_ls if i not in sw]
    return word_ls


def build_save_f_name(path, n='_default'):
    name = path.split(" ")[0]
    return name + "_" + n
    

def remove_punctuation(raw_str, to_blank=False):
    # punctuation = r"~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./\n"
    punctuation = r"~@#%^&*_+`{}|\[\]\";\-\\\='<>/\n"
    sub_str = ""
    if to_blank is True:
        sub_str = " "
    res = re.sub(r'[{}]+'.format(punctuation), sub_str, raw_str.lower())
    return res


def replace_tags(text):
    dr = re.compile(r'<[^>]+>', re.S)
    dd = dr.sub('<>', text)
    return dd


def remove_repeat_with_order(x, ban_none=True, ban_space=True):
    res = []
    for e in x:
        if e not in res:
            if ban_none and e == '':
                continue
            if ban_space and e == ' ':
                continue
            res.append(e)
    return res


def parse_content(url, content, ticket_id, country_code, origanization_id, data_type):
    content_low = content.lower()
    # 用户投诉句子：content， 用户投诉句子(小写)：content_low
    # 关键词匹配
    # 在下面描述发生了什么, 投诉说明, 描述通话请求, 在下面描述， 主要信息,  描述
    kw = ['descrição', 'descreva o ocorrido', 'estou tendo problema a dias', 'descreva abaixo', 'mais informações', 'descreva']
    temp_kw = ''
    flag = False
    for k in kw:
        if k in content_low:
            temp_kw = k
            flag = True
            break
    if flag:
        kw_idx = content_low.index(temp_kw)
        complaint = replace_tags(content[kw_idx:]).replace('\n', ' ')
        cpl = re.split('<>|:', complaint)
        cpl = remove_repeat_with_order(cpl)
        try:
            cpl_len = len(cpl[1].split(" "))
            if cpl_len < 5: # 句子长度小于5不需要
                print('小于5')
                return None
            # trans_obj = trans.translate(cpl[1], dest='zh-CN', src='pt')
            return str(ticket_id) + "|" + str(country_code) + "|" + str(origanization_id) + "|" + cpl[1] + "|" + "翻译位置" + "|" + str(data_type)
        except Exception as e:
            print(e)
            return None
    else:
        res_no_pun = remove_punctuation(content_low)
        print(res_no_pun)
        if res_no_pun == content_low:
            # trans_obj = trans.translate(content_low, dest='zh-CN', src='pt')
            return str(ticket_id) + "|" + str(country_code) + "|" + str(origanization_id) + "|" + content_low + "|" + "翻译位置" + "|" + str(data_type)
    print("没命中")
    return None


def get_content(ticket_id, country_code, origanization_id, data_type, ch=5, req='passenger'):
    # ticket_id = line['ticket_id']
    # country_code = line['country_code']
    # origanization_id = line['organization_id']
    url = "http://10.14.128.18:8000/cs/ark/service/ticket/findTicketById?organization_id={}&canonical_country_code={}&lang=en&id={}" \
            .format(origanization_id, country_code, ticket_id)
    
    response = requests.get(url)
    response_d = json.loads(response.text)
    try:
        channel = response_d["data"]["ticket"]["channelType"]
        requester = response_d["data"]["ticket"]["requesterTypeName"]
        content = response_d["data"]["ticket"]["content"]
    except BaseException as e:
        print(e)
        return
    if channel != ch and requester != req: # 选取来自channel和requester的句子
        print('wrong channle or wrong requester')
        return
    res = parse_content(url, content, ticket_id, country_code, origanization_id, data_type)
    return res


"""
通过请求工单url, 根据 organization_id, country_code, ticket_id获取工单内容
然后对工单内容进行解析, 获取投诉哦内容, 将解析后的投诉内容存入文本
"""
def get_p_n_content(path):
    f_read = open(path, 'r', encoding='utf-8')
    save_path_no_trip = build_save_f_name(path, 'no_trip.csv')
    save_path_other = build_save_f_name(path, 'others.csv')

    # 第一次运行的时候打开创建, 如果中间出现
    # f_save_no_trip = open(save_path_no_trip, 'w', encoding='utf-8')
    # f_save_others = open(save_path_other, 'w', encoding='utf-8')
    # head = f_read.readline().replace(',', '|')
    # f_save_no_trip.write(head)
    # f_save_others.write(head)

    for i, line in enumerate(f_read.readlines()[1216315:]): # 在这里可以设置截断位置
        line = line.replace('"','').replace(',', '|')
        line_ls = line.split('|')
        print(line_ls)
        ticket_id = line_ls[0]
        country_code = line_ls[2]
        origanization_id = line_ls[3]
        data_type = line_ls[-1]

        # 跳过 not_fee_cate 的投诉, 原因是拿不到
        if data_type == 'not_fee_cate\n':
            continue

        if i % 100 == 0:
            print("time record: {}".format(datetime.datetime.now().microsecond))

        if data_type == 'no_trip\n':
            print('no_trip')
            res = get_content(ticket_id, country_code, origanization_id, data_type)
            with open(save_path_no_trip, 'a', encoding='utf-8') as f:
                if res != None:
                    f.write(res)
                else:
                    print('?')
        else:
            res = get_content(ticket_id, country_code, origanization_id, data_type)
            with open(save_path_other, 'a', encoding='utf-8') as f:
                if res != None:
                    f.write(res)
                else:
                    print('??')


"""
读取文件内容, 通过请求词向量url将文本内容转为句向量(+padding), 并存储为npy文件
"""
def request_model_inter(path):
    result = []
    f = open(path, 'r')
    f.readline() # jump
    for i, line in enumerate(f.readlines()):
        line_ls = line.split('|')
        # 主要用了 content 和 ticket_id 两个参数, 根据文本可以自行调整
        content = line_ls[2]
        ticket_id = line_ls[0]

        # 测试用暂停, -1换成其他数
        if i == -1: break

        print("step: %d, ticket_id: %s" % (i, ticket_id))
        content_d["text"] = content
        content_str = json.dumps(content_d)
        reponse = requests.post(url, content_str, header)
        res = json.loads(reponse.text)
        try:
            wv_with_padding = np.array(res['data']['word_vec_padding'])
            wv_with_padding_fat = wv_with_padding.flatten()
            result.append(wv_with_padding_fat.tolist())
        except Exception as e:
            print(e)
    res_arr = np.array(result)
    save_path = build_save_f_name(path[:-5], "len-100_mat.npy")
    # save
    np.save(save_path, res_arr)


"""
测试用, 输入单行数据, 将单行数据转为句向量
"""
def request_single_simple(line):    
    line_ls = line.split('|')
    content = line_ls[1].replace('\n', '')
    ticket_id = line_ls[0]
    content_d["text"] = content
    content_str = json.dumps(content_d)
    reponse = requests.post(url, content_str, header)
    res = json.loads(reponse.text)
    try:
        wv_with_padding = np.array(res['data']['word_vec_padding'])
        print(wv_with_padding.shape)
        wv_with_padding_fat = wv_with_padding.flatten()
        wv_with_padding_mean = sum(wv_with_padding) / len(wv_with_padding)
        return wv_with_padding_fat, wv_with_padding_mean, ticket_id, content # 打平后的句向量, 均值句向量, 订单id, 文本原句
    except Exception as e:
        print("找不到特定关键字")
        print(e)


"""
根据本地词向量模型获取句向量, 并存入为npy文件
"""
def request_model_inter_local(path, w2v_path):
    sent_length = 30
    f = open(path, 'r')
    f.readline() # jump
    model = KeyedVectors.load_word2vec_format(w2v_path)
    ready2 = []
    for idx1, line in enumerate(f.readlines()):
        line_ls = line.split('|')
        content = line_ls[3]
        ticket_id = line_ls[0]
        if idx1 == -1:
            break
        print("step: %d, ticket_id: %s" % (idx1, ticket_id))
        content_ls = preprocess_data(content)
        if len(content_ls) < sent_length:
            content_ls += ['<PAD>' for i in range(sent_length - len(content_ls))]
        else:
            content_ls = content_ls[:sent_length]
        dim = 0
        ready = []
        for w in content_ls:
            if w in model:
                wv = model[w]
                dim = len(wv)
                ready.append(wv)
            else:
                wv = np.array([0. for i in range(dim)])
                ready.append(wv)
        res = np.concatenate(ready, axis=0)
        if res.shape[0] != 3150:
            print('error shape, jump')
            continue
        res = res.reshape(1, -1)
        ready2.append(res)

    total_vec = np.concatenate(tuple(ready2), axis=1).reshape(-1, ready2[0].shape[1])
    print(total_vec.shape)
    save_path = build_save_f_name(path, "_basew2v6_len-" + str(sent_length) + "_mat.npy")
    np.save(save_path, total_vec)



"""
测试用, 根据一行测试数据从词向量模型中获取句向量 
"""
def request_single_simple_local(line, model, sent_length):
    line_ls = line.split('|')
    content = line_ls[3]
    ticket_id = line_ls[0]
    content_ls = preprocess_data(content)
    #
    if len(content_ls) < sent_length:
        content_ls += ['NAN' for i in range(sent_length - len(content_ls))]
    else:
        content_ls = content_ls[:sent_length]
    dim = 0
    ready = []
    for idx2, w in enumerate(content_ls, start=1):
        if w in model and w != 'NAN':
            wv = model[w]
            dim = len(wv)
            ready.append(wv)
        else:
            wv = np.array([0. for i in range(dim)])
            ready.append(wv)
    res = np.concatenate(ready, axis=0)
    #
    if res.shape[0] != 3150:
        # print(res.shape[0])
        print('error shape, jump')
        return "?", "?", "?", "?"
    #
    wv = res.reshape(1, -1) #.astype(np.double)
    wv_with_padding_mean = sum(ready) / len(ready)
    return wv, wv_with_padding_mean, ticket_id, content




if __name__ == "__main__":
    path1 = sys.argv[1]
    # path2 = sys.argv[2]

    # 获取投诉文本
    # get_p_n_content(path1)

    # 正负投诉样本 -> 矩阵文件
    # request_model_inter_local(path1, path2)
    
    # 单行测试(url)
    line = "360287970317045800|Motorista pegou o passageiro errado e cobrou do meu cartão|0|0|司机接错了乘客并向我收费|[[0.57934713 0.42065284]]|乘客未上车"
    v,_,_,_ = request_single_simple(line=line)  
    print(v.tolist())

    # 单行测试(本地)
    w2v_model = KeyedVectors.load_word2vec_format(path1)
    v,_,_,_ = request_single_simple_local(line, w2v_model, sent_length=30)
    print(v,tolist())