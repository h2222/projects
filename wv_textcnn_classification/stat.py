
import sys, os
import re
import json

from tqdm import tqdm
import pandas as pd

from nltk.corpus import stopwords
from nltk import tokenize
from gensim.models import KeyedVectors 
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.callbacks import CallbackAny2Vec
from googletrans import Translator
from collections import OrderedDict

sw = stopwords.words('portuguese')


def preprocess_data(content):
    word_ls =  tokenize.word_tokenize(content, language='portuguese') 
    word_ls = [i for i in word_ls if i not in sw]
    word_sp_sent = ' '.join(word_ls)
    return word_sp_sent


def stat_frequence(path):
    df = pd.read_csv(path, sep='|', encoding='utf-8')
    stat = {}
    for sent in tqdm(df['content']):
        # print(sent)
        word_list = preprocess_data(sent).split(' ')
        for w in word_list:
            if w not in stat:
                stat[w] = 0
            stat[w] += 1
    
    stat =list(sorted(stat.items(), key=lambda x : x[-1], reverse=True))[:150]

    rg = '('
    for k, v in stat:
        rg += k+'|'
    rg += ')'
    print(rg)

"""
用规则的方式区分:上车 和 未上车 样本
"""
def new_split_dataset(path):
    # v1
    # rg1 = r'(没|没有|没来|没有来|没让我|没让|没想|拒绝|未|无法)+(去|来|上|上车|到|参加|找|找到|接|接我|出现|发生|露面|等|等我|抓|登|坐|坐上|我|带|人|跟|完成|回答)'
    # rg2 = r'(不|不想|不能|不愿意|不允许|不愿|不会|不肯|拒绝|未|无法)+(去|来|在|带|到|等待|取消|参加|出现|上车|发生|露面|使用|记得|找|坐|跟|完成|回答)'
    # rg3 = r'(抓|登|上|走)+错'
    # rg4 = r'(错|错误|错误的|错过了|其他的|其他|其他的|另外|另外的|另一个|找不到|看不到)+(乘客|人|地方|地址|地点|起点|位置|放置)'
    # rg5 = r'(等待|等)'

    # v2
    # no_trip regax
    rg1 = r'(没|没有|没来|没有来|没让我|没让|没想|拒绝|未|尚未|无法|弃了)+(开始|进行|去|来|上|上车|到|参加|找|找到|等|接|接我|当场|出现|发生|露面|等|等我|抓|登|坐|坐上|我|他|带|人|跟|完成|回答|乘客|司机|分钟|取消|按要求|要求|收到|这次|靠近|停下|订购|搭车\
        |下令|说明|停|出发|看见我|比赛|跑|车|举行|旅行|没抓|用车|和我一起|联系|及时)'
    rg2 = r'(不|不是|不是为了|不为了|不想|不能|不愿意|不允许|不愿|不会|不肯|拒绝|未|无法)+(走|我|他|去|来|在|带|到|等|等待|取消|参加|出现|上车|发生|露面|使用|记得|找|坐|跟|完成|回答|乘客|司机|取消|让我|看到|搭车|登机)'
    rg3 = r'(抓|抓到|抓到了|登|上|走)+(错|其他人|别人|其他)'
    rg4 = r'不知道+(这次|那次|这场|那场|这个|发生)'
    rg5 = r'(错|错误|错误的|正确|正确的|错过了|其他的|其他|其他的|另外|另外的|另一个|另一位|找不到|看不到|取消了|取消|离开了)+(乘客|人|地方|地址|地点|起点|位置|放置|比赛|旅行|我|街道)'
    rg6 = r'(等待|等|代替|不认识|接错)'
    rg7 = r'(司机|驾驶员|他|那家伙)+(没有替|独自|没看到|取消|结束|不是|不想|不能|不愿|不允许|不会|不肯|拒绝|未|无法|没来|没有来|没有到|没有赶|没让我|没让|没想|拒绝|未|无法|不希望)'
    rg9 = r'(等待|等|代替|不认识|接错|乘客错了|接我之前|司机没来|我到了, 司机没有|位置不对)'
    rg10 = r'(我|司机|他|司机|那家伙)+(甚至没有|接了|没有比赛|没要比赛|没比赛|没要求|联系|把别人|钉错|没有点|离开|没有和|取消|已经取消|没有出现|没有出行|和其他人|和另|和别人|不知道谁|不知道是谁|带别|带其他|带另|去了别|去了其|去了另\
        |错乘|没有比赛|没要比赛|没有做)'
    # trip regax
    # rg7 = r'^[如果可退款将通过折扣券]'
    rg8 = r'(没|没有|没来|没有来|没让我|没让|没想|拒绝|未|无法|尚未)+(开始|进行|去|来|上|上车|到|参加|找|找到|接|接我|出现|发生|露面|等|等我|抓|登|坐|坐上|乘客|司机|分钟|取消|按要求|要求|这次|靠近|停下|在那)'
   

    # 西班牙语 regex test1 (效果不好, 作废)
    rg_p_total = r'((cancel)(.*?)( solicit(.*?)| corrida))|((sem)(.*?)( mi(.*?)| final(.*?)| passag(.*?)))|((mot(.*?)|ele)+( recus(.*?)| não most(.*?)| não veiu| não fez| cancel(.*?)| nao encer(.*?)| deu| não quis| embarcou| pegou))|((não|nao|não quis)+( est| vem| veio| vêm| cheg(.*?)| embar(.*?)| fiz| apare(.*?)| acont(.*?)| encontr(.*?)| solicit(.*?)| entr(.*?)| me bus(.*?)| final(.*?)| realizar)|((não)(.*?)( essa| feita| no carro)))|(no carro)|(quis fazer a corrida)'
    rg_p1=  r'(cancel)(.*?)( solicit(.*?)| corrida)'
    rg_p2 = r'(sem)(.*?)( mi(.*?)| final(.*?)| passag(.*?))'
    rg_p3 = r'(mot(.*?)|ele)+( recus(.*?)| não most(.*?)| não veiu| não fez| cancel(.*?)| nao encer(.*?)| deu| não quis| embarcou| pegou)'
    rg_p4 = r'(não|nao|não quis)+( est| vem| veio| vêm| cheg(.*?)| embar(.*?)| fiz| apare(.*?)| acont(.*?)| encontr(.*?)| solicit(.*?)| entr(.*?)| me bus(.*?)| final(.*?)| realizar)'
    rg_p5 = r'(não)(.*?)( essa| feita| no carro| veio)'
    rg_p6 = r'(no carro|quis fazer a corrida)'

    df = pd.read_csv(path, sep='|', header=None, encoding='utf-8')
    df.columns = ["ticket_id", "country_code", "organization_id", "content", "trans", "category_id"]
    df = df.loc[df['trans'] != "如果可退款，将通过折扣券。"]
    df_p = pd.DataFrame(columns=df.columns)
    df_n = pd.DataFrame(columns=df.columns)
    for rg in [rg1, rg2, rg3, rg4, rg5, rg6, rg7, rg9, rg10]:
        df_target = df.loc[df['trans'].str.extract(rg).dropna().index]
        print(df_target.index)
        df = df[~ df.index.isin(df_target.index)]
        df_p = df_p.append(df_target, ignore_index=False)
    # df_p 正样本, df 负样本
    df_p.to_csv(path+'_rg_p.csv', index=False, sep='|', encoding='utf-8-sig')
    df.to_csv(path+'_rg_n.csv', index=False, sep='|', encoding='utf-8-sig')


"""
把.vec.json 转化为.vec文件
"""
def w2v_json2vec(path):
    f = open('test/w2v.vec', 'w', encoding='utf-8')
    f.write('150698 105\n')
    with open(path, 'r', ) as f:
        w2v = json.load(f)
        for k in tqdm(w2v):
            vec = w2v[k]
            vec = list(map(str, vec))
            vec_str = ' '.join(vec)
            res = k + ' ' + vec_str + '\n'
            # print(res)
            # break
            with open('test/w2v.vec', 'a', encoding='utf-8') as f2:
                f2.write(res)


"""
词向量txt转二进制文件bin
"""
def vec2bin(path):
    name = path.split('.')
    name.pop()
    # print(name[0])
    bin_path = name[0] + '_b.bin'
    model_vec = KeyedVectors.load_word2vec_format(path)
    print(model_vec.vectors.shape)
    print("vec result:\n"+ str(model_vec["caminho"]))
    model_vec.save_word2vec_format(bin_path, binary=True)
    model_bin = KeyedVectors.load_word2vec_format(bin_path, binary=True)
    print("bin result:\n"+ str(model_bin["caminho"]))


"""
导入w2v文件, 然后进行相似排序
"""
def load_w2v(path):
    model = KeyedVectors.load_word2vec_format(path)
    print(len(model.vectors))

    for s in ['participei', 'compareceu', 'que', 'confirmação', 'cancelar']:
        try:
            sim_top_20 = model.most_similar(s, topn=20)
            print('与 "%s" 相关词有\n' % s)
            for k in sim_top_20:
                print('%s %.3f' % (k[0], k[1]))
        except Exception as e:
            print('error:', e)
        print('-------------------')


"""
将文本预处理:
导入正负样本文件
例如: 我爱北京天安门. -> 我 爱 北京 天安门 .
每行一句话, 去除停止词, 空格分词, 保留标点
"""
def preprocess(path1, path2, save_path):
    pos_df = pd.read_csv(path1, sep='|', encoding='utf-8')
    neg_df = pd.read_csv(path2, sep='|', encoding='utf-8')
    pos_df.columns = ["ticket_id", "country_code", "organization_id", "content", "trans", "category_id"]
    neg_df.columns = ["ticket_id", "country_code", "organization_id", "content", "trans", "category_id"]

    X_train = []
    for content in tqdm(pos_df['content']):
        try:
            word_ls =  tokenize.word_tokenize(content, language='portuguese') 
            word_ls = [i for i in word_ls if i not in sw]
            word_sp_sent = ' '.join(word_ls)
        except Exception as e:
            print(e)
            continue
        X_train.append(word_sp_sent+'\n')
        
    for content in tqdm(neg_df['content']):
        try:
            word_ls =  tokenize.word_tokenize(content, language='portuguese') 
            word_ls = [i for i in word_ls if i not in sw]
            word_sp_sent = ' '.join(word_ls)
        except Exception as e:
            print(e)
            continue
        X_train.append(word_sp_sent+'\n')

    with open(save_path, 'w', encoding='utf-8') as f:
        f.writelines(X_train)

    return save_path


def add_line(path, temp_path):
    with open(path, 'r') as f:
        for line in f.readlines():
            line_list = line.split('|')
            if len(line_list) != 5:
                break
            ticket_id = line_list[0]
            print(ticket_id)
            country_code = line_list[1]
            organization_code = line_list[2]
            content = line_list[3]
            category_id = line_list[4]
            new_line = ticket_id + '|' + country_code + '|' + organization_code + '|' + content + '|'+ '翻译位置' + '|' + category_id
            with open(temp_path, 'a', encoding='utf-8') as f2:
                f2.write(new_line) 

"""
callback函数, 返回word2vec的loss, epoch信息
"""
class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


"""
训练词向量模型, 保存vec(可读)和bin(不可读)两个版本
"""
def train_w2v(path):
    model = Word2Vec(LineSentence(path), iter=40, size=105, window=3, min_count=1, compute_loss=True, callbacks=[callback()])
    model.wv.save_word2vec_format('test/save_wv/w2v_v6.vec', binary=False)
    model.wv.save_word2vec_format('test/save_wv/w2v_v6_b.bin', binary=True)



"""
将西班牙语翻译为中文
"""
def trans(path, start, end):
    tran_path = 'test/trans_content/45910_trip_trans_start-%d_end-%d.csv' % (start, end)
    if not os.path.exists(tran_path):
        os.system('touch %s' % tran_path)
    translator = Translator()
    f = open(path, 'r')
    context = f.readlines()[start:end]
    for line in context:
        line_ls = line.split('|')
        content = line_ls[3]
        try:
            trans_obj = translator.translate(content, dest='zh-CN', src='pt')
            trans_content = trans_obj.text
        except Exception as e:
            print(e)
            print("翻译异常")
            trans_content = "翻译异常"
        line_ls[4] = trans_content
        line = '|'.join(line_ls)
        print(line)
        with open (tran_path, 'a', encoding='utf-8') as f2:
            f2.write(line)




if __name__ == "__main__":
    path1 = sys.argv[1]
    # path2 = sys.argv[2]
    # start = int(sys.argv[2])
    # end = int(sys.argv[3])

    # path2 = sys.argv[2]
    # new_split_dataset(path1)

    # stat_frequence(path1)

    # w2v_json2vec(path)
    # load_w2v(path1)

    # 临时添加一行
    # add_line(path1, path2)


    # save_sent_path = preprocess(path1, path2)
    # train_w2v(save_sent_path)

    # 翻译
    # trans(path1, start, end)

    # vec 文件转 bin
    vec2bin(path1)




