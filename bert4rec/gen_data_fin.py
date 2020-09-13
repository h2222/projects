import sys, os
import codecs
import random
import copy
import multiprocessing
from multiprocessing import Pool, cpu_count
import time
import collections
from collections import defaultdict
import numpy as np
import tensorflow as tf
import six
import pickle

from vocab import FreqVocab

MaskedLmInstance = collections.namedtuple('MaskedLmInstance', ['index', 'label'])


#%%
def data_partition(path):
    '''
    数据集格式 : user item   都为数字, 空格分隔, 代表不同的用户选择不同的商品
    例如          1    2
                  1    3
                  2    19
                  ...
    '''
    user_num = 0
    item_num = 0
    user = defaultdict(list) # 默认dict, 如果key不存在返回对应类型, 例如[]
    user_train = {}
    user_test = {}
    user_valid = {}

    f = open(path, 'r')

    for line in f.readlines():
        u, i = line.strip().split(' ')     # strip()去两头, rstrip()去右边, lstrip()去左边
        u, i = int(u), int(i)
        user_num = max(user_num, u)
        item_num = max(item_num, i)
        user[u].append(i)
    
    for u in user:
        nfeedback = len(user[u])
        # 顾客点击的商品少于三个, 没法分配测试集和预测集
        if nfeedback < 3:
            user_train[u] = user[u]
            user_valid[u] = []
            user_test[u] = []
        else:
            # 顾客点击商品大于3个, [-2]为测试集, [-1]为预测集, 其他的为训练集
            user_valid[u] = []
            user_test[u] = []
            user_train[u] = user[u][:-2]
            user_valid[u].append(user[u][-2])
            user_test[u].append(user[u][-1])

    return [user_train, user_valid, user_test, user_num, item_num]

#%%

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list = tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list = tf.train.FloatList(value=list(values)))
    return feature


def write_instance_to_example_files(instances,
                                    max_seq_length,
                                    max_predictions_per_seq,
                                    vocab,
                                    output_files):
    """
    从 TrainingInstance 创建一个TF example file 
    """

    writers = []
    writer_index = 0
    total_written = 0
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))


    for (inst_index, instance) in enumerate(instances):
        try:
            # 通过tokens序列, 转化为 ids
            # instance.tokens 表示1. 随机mask的token list, 2. 只有尾部为mask的token序列
            input_ids = vocab.tokens2ids(instance.tokens)
        except:
            print('something wrong....')
            print(instance)
    
        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length  ## 输入序列必须小于最大序列长度
        
        input_ids += [0] * (max_seq_length - len(input_ids)) # input_ids不够max_seq_len, 补0
        input_mask += [0] * (max_seq_length - len(input_mask)) # Mask 范围仅限于input_ids长度, 其他地方不mask
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        # [mask 商品的位置]
        masked_lm_positions = list(instance.masked_lm_positions)
        # [mask 商品的 ids, 编号]
        masked_lm_ids = vocab.tokens2ids(instance.masked_lm_labels)
        # 初始化masked权重为1
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        ## 长度不够补0
        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        # 建立对应特定类型的特征, 病存储在features有序字典中
        features['info'] = create_int_feature(instance.info)  # user ids(1)
        features['input_ids'] = create_int_feature(input_ids) # 输入序列的长度 (最大50)
        features['input_mask'] = create_int_feature(input_mask)# 可以被mask的长度最大(50)
        features['input_lm_positions'] = create_int_feature(masked_lm_positions) # 被mask的位置index(最大30, 因为mask rate=0.6)
        features['input_lm_ids'] = create_int_feature(masked_lm_ids) # 被mask位置的ids(30长)
        features['input_lm_weight'] = create_float_feature(masked_lm_weights) # 被mask的权重(30长)

        # 使用tf.train.Features 建立多种类型的特征集
        # 使用 tf.train.Example 创建样本, example对应一个样本
        tf_example = tf.train.Example(features = tf.train.Features(feature=features))
        

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index+1) % len(writers)
        total_written += 1

        if inst_index < 20:
            print('*** example ***')
            print('tokens: %s' % ' '.join([str(x) for x in instance.tokens]))

            for feature_name in features:
                feature = features[feature_name]
                values = []
                ## 判断 features 类型, 并打印对应list
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                print('%s:%s' % (feature_name, ' '.join([str(x) for x in values])))

    for writer in writers:
        writer.close()
    print('wrote %d total instances', total_written)

            




#%%

def create_instances_threading(all_documents,
                              user,
                              max_seq_length,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              vocab,
                              rng,
                              mask_prob,
                              step):
    '''
    单一进程任务
    '''
    cnt = 0
    start_time = time.clock()
    instances = []
    for user in all_documents:
        # 测试早停用
        #if cnt == 20:
        #    break
        cnt += 1
        if cnt % 1000 == 0:
            print('step:{}, name:{}, step:{}, time:{}'.format(
                step, 
                multiprocessing.current_process().name,
                cnt,
                time.clock() - start_time))

        #  instance list
        instances += create_instances_from_document_train(all_documents=all_documents,
                                                          user=user,
                                                          max_seq_length=max_seq_length,
                                                          short_seq_prob=short_seq_prob,
                                                          masked_lm_prob=masked_lm_prob,
                                                          max_predictions_per_seq=max_predictions_per_seq,
                                                          vocab=vocab,
                                                          rng=rng,
                                                          mask_prob=mask_prob)
    
    return  instances


#%%
class TrainingInstance:
    """
    句子对训练实例
    A single traing instance (sentence pair)
    """

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info # [用户名称]
        self.tokens = tokens # [带mask的, 该用户点击的商品]
        self.masked_lm_positions = masked_lm_positions # [被mask的位置]
        self.masked_lm_labels = masked_lm_labels # [被mask前商品名称]
    
    def __str__(self):
        '''
        一些训练实例信息供打印使用
        '''
        s = ''
        s += 'info: %s\n' % (' '.join([str(i) for i in self.info]))
        s += 'tokens: %s\n' % (' '.join([str(i) for i in self.tokens]))
        s += 'masked_lm_positions: %s\n' % (' '.join([str(i) for i in self.masked_lm_positions]))
        s += 'masked_lm_lables: %s\n' % (' '.join([i for i in self.masked_lm_labels]))
        s += '\n'
        return s
    
    def __repr__(self):
        return self.__str__()
        


#%%

def create_masked_lm_predictions_force_last(tokens): # item list [12, 32, 55, 63, ..]
    """
    为 Training instance 对象提供必要的参数
    包括: tokens 序列(ids 序列),  掩盖位置([ids]), 掩盖的label([label])
    """
    last_index = -1
    # 最后一个token(预测位置)的index定位
    for (i, toke) in enumerate(tokens):
        if tokens == '[CLS]' or tokens == '[PAD]' or tokens == '[NO_USE]':
            continue
        last_index = i
    
    assert last_index > 0

    # 保证 tokens 为list, 并在最后位置变为 [MASK]
    output_tokens = list(tokens)
    output_tokens[last_index] = '[MASK]'

    masked_lm_positions = [last_index]
    masked_lm_labels  = [tokens[last_index]]  # 没变[MASK]之前, 最后一个的tokens

    return (output_tokens, masked_lm_positions, masked_lm_labels)




def create_instance_from_document_test(all_documents, # {'user_id':[['抽样的item_ids']]}
                                       user,
                                       max_seq_length):
    """
    创建一个 training instance, 该实例中包含必要的参数(是一个op类)
    """
    document = all_documents[user]
    max_num_tokens = max_seq_length
    tokens = document[0]
    
    assert len(document) == 1 and len(document[0]) <= max_num_tokens # 商品序列==1 且 序列长度<max
    assert len(tokens) >= 1 #商品数>=1

    (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

    info = [int(user.split('_')[1])]

    instance = TrainingInstance(info=info,
                                tokens=tokens,
                                masked_lm_positions=masked_lm_positions,
                                masked_lm_labels=masked_lm_labels)

    return [instance]




def mask_last(all_documents,
              user,
              max_seq_length,
              short_seq_prob,
              masked_lm_prob,
              max_predictions_per_seq,
              vocab,
              rng):
    """
    遮盖序列的最后一个token
    """
    document = all_documents[user]
    max_num_tokens = max_seq_length
    instances = []

    # info:顾客编号, vocab_item
    info = [int(user.split('_')[1])]

    for tokens in document:
        assert len(tokens) <= len(tokens) <= max_num_tokens

        # 遮盖最后一个, 返回tokens 序列, 遮盖位置, 和遮盖前label
        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)

        instance = TrainingInstance(info=info,
                                    tokens=tokens,
                                    masked_lm_positions=masked_lm_positions,
                                    masked_lm_labels=masked_lm_labels)
    
        instances.append(instance)

    return instances






def create_masked_lm_predictions(tokens,
                                masked_lm_prob,
                                max_predictions_per_seq,
                                vocab_words,
                                rng,
                                mask_prob):
    
    """
    返回被mask过的序列, mask的位置, 和mask前的值
    """

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token not in vocab_words:  # vocab_words 所有商品牌的 list
            continue
        cand_indexes.append(i) # 加入候选token ids
    
    rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    
    ## 需要预测的item的格式, 最不超过30 个, 至少1个
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens)*masked_lm_prob))))
    
    # 
    masked_lms = []
    convered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in convered_indexes:
            continue
        convered_indexes.add(index)
        
        masked_token = None
        # 
        # 80% 使用 [MASK] 代替
        if rng.random() < mask_prob:
            masked_token = '[MASK]'
        # 10% 保持原值
        else:
            if rng.random() < 0.5:
                masked_token = tokens[index]
            else:
            # 10% 随机选取一个token 代替
                masked_token = rng.choice(vocab_words) 

        # 将原位置替换, 改变, 或不变
        output_tokens[index] = masked_token

        # masked_lms 结构 [{name='MaskedLmInstance', 'tokens的index':'xxxx', 'tokens的值':'xxxxx'}, {....}, .....]
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x:x.index) # 根据index对masked_lms进行排序

    masked_lm_positions = []
    masked_lm_labels = []

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    
    # 返回被mask过的序列, mask的位置, 和mask前的值
    return (output_tokens, masked_lm_positions, masked_lm_labels)



def create_instances_from_document_train(all_documents,
                                        user,
                                        max_seq_length,
                                        short_seq_prob,
                                        masked_lm_prob,
                                        max_predictions_per_seq,
                                        vocab,
                                        rng,
                                        mask_prob):
    """
    一个训练任务
    """
    document = all_documents[user] # [['采样商品的ids(频率)'], [....]]
    max_num_tokens = max_seq_length

    instances = []
    info = [int(user.split('_')[1])] # 用户 ids
    vocab_items = vocab.get_items() #  [所有商品的list]


    for tokens in document:
        assert 1 <= len(tokens) <= max_num_tokens  
        (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                                                           tokens=tokens, 
                                                           masked_lm_prob=masked_lm_prob,
                                                           max_predictions_per_seq=max_predictions_per_seq,
                                                           vocab_words=vocab_items,
                                                           rng=rng,
                                                           mask_prob=mask_prob)
        ## 创建训练instance
        instance = TrainingInstance(info=info,
                                    tokens=tokens,
                                    masked_lm_positions=masked_lm_positions,
                                    masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    ## 返回训练用 instances list
    return instances





#%%

def create_training_instance(all_documents_raw, # {'user_123':[[点击商品的ids]]}
                             max_seq_length,
                             dupe_factor,
                             short_seq_prob,
                             masked_lm_prob,
                             max_predictions_per_seq,
                             rng,
                             vocab,
                             mask_prob,
                             prop_sliding_window,
                             pool_size,
                             force_last=False):
    
    """
    将 raw data 变为 可被 estimator接受的 training 实例 
    """
    all_documents = {} # {'user_id':[['采样的商品id']]}
    # force_last 注意最后一个, 必须取全量长度序列
    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print('got empty seq:'+user)
                continue
            # 选取前 max_num_tokens 个 token 的 ids 
            all_documents[user] = [item_seq[-max_num_tokens:]] 

    else:
        max_num_tokens = max_seq_length
        # prop_sliding_window 指在训练中将部分输入作为训练数据的比例
        # max_num_tokens 指序列(tokens)总长度
        # cloze task:(原文) Cloze task(also known as “Masked Language
        # Model”) to sequential recommendation. It is a test consisting
        # of a portion of language with some words removed，where the
        # participant is asked to fill the missing words
        sliding_step = (int)(prop_sliding_window * max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
        

        ## 采样步骤
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print('got empty seq:'+user)
                continue
            # 添加padding
            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                # item idx 长度超过 max_len, 按照采样大小将数据划分成大小为 sliding_step的段
                # 为了可以均匀采样
                beg_idx = [i for i in range(len(item_seq)-max_num_tokens, 0, -sliding_step)]
                beg_idx.append(0) # 最后一位添0
                all_documents[user] = [item_seq[i:i+max_num_tokens] for i in beg_idx[::-1]] # 因为 beg idx 是反的               

    instances = []
    if force_last:
        for user in all_documents:
            # instance 是一个 list, 里面是不同用户的 TrainingInstance信息
            # [TrainingInstance1, TrainingInstance2, TrainingInstance3,  ...]
            instances += create_instance_from_document_test(all_documents=all_documents,
                                                            user=user,
                                                            max_seq_length=max_seq_length)
        print('num of instance: %s' % (len(instances)))
    else:
        start_time = time.clock()
        pool = Pool(pool_size)
        instances = []
        print('document num: {}'.format(len(all_documents)))
        

        # 结果记录子函数
        def log_result(result):
            print('callback function result type: {}, size: {}'.format(type(result), len(result)))
            instances.extend(result)
        
        for step in range(dupe_factor):
            ## applys_async(func_A, args_A, func_B)
            # 调用 带有参数A函数, 将 A函数的结果传入B函数
            pool.apply_async(
                create_instances_threading, args=(all_documents,
                                                  user,
                                                  max_seq_length,
                                                  short_seq_prob,
                                                  masked_lm_prob,
                                                  max_predictions_per_seq,
                                                  vocab,
                                                  random.Random(random.randint(1, 10000)),
                                                  mask_prob,
                                                  step),
                                            callback=log_result)
            
        pool.close()
        pool.join()
        # isntances list, 里面是 TrainingInstance的实例对象, 每个对象包含三个变量
        # instances 的长度表示 step * 顾客数量
        # TrainingInstance对象的参数包括: 
        # 1.info 用户编号, 2. 带mask的tokens list
        # 3. 被mask的位置, 4. masked前原有的items
        # print('-------', len(instances))

        for c, user in enumerate(all_documents):
            # 实验早停
            #if c == 20:
            #    break
            ## instance list 表示(只有尾部!!!)token被[MASK] TrainInstance对象序列
            instances += mask_last(all_documents=all_documents,
                                   user=user,
                                   max_seq_length=max_seq_length,
                                   short_seq_prob=short_seq_prob,
                                   masked_lm_prob=masked_lm_prob,
                                   max_predictions_per_seq=max_predictions_per_seq,
                                   vocab=vocab,
                                   rng=rng)
            
            print('num of instance:{}; time:{}'.format(len(instances), time.clock()-start_time))

        
    return instances 
    ##############################  上次位置  ############################



#%%

def gen_samples(data, # {'user_123':[[点击商品的ids]]}
                output_fname,
                rng,
                vocab,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                mask_prob,
                masked_lm_prob,
                max_predictions_per_seq,
                prop_sliding_window,
                pool_size,
                force_last=False):
    
    # create train
    instances = create_training_instance(all_documents_raw=data,
                                        max_seq_length=max_seq_length,
                                        dupe_factor=dupe_factor,
                                        short_seq_prob=short_seq_prob,
                                        masked_lm_prob=masked_lm_prob,
                                        max_predictions_per_seq=max_predictions_per_seq,
                                        rng=rng,
                                        vocab=vocab,
                                        mask_prob=mask_prob,
                                        prop_sliding_window=prop_sliding_window,
                                        pool_size=pool_size,
                                        force_last=force_last)
    
    ## 记录日志
    print('*** Writer to output file ***')
    print(' %s', output_fname)
    
    ## 单个instanace 是 TrainingInstance 的实例


    # 将数据写入 TFrecord 文件中
    write_instance_to_example_files(instances,
                                    max_seq_length,
                                    max_predictions_per_seq,
                                    vocab,
                                    [output_fname])
    
    

#%%

def main():
    ## 参数 params
    max_seq_length = 50
    max_predictions_per_seq = 30
    masked_lm_prob = 0.6 # 已知item mask
    mask_prob = 1.0 # 预测 item mask
    dupe_factor = 10 # 10
    prop_sliding_window = 0.1
    pool_size = 10 # 10
    random_seed = 666
    short_seq_prob = 0 # Probability of creating sequences which are shorter than max seq length
    rng = random.Random(random_seed) # random number generation

    output_dir = './data/' 
    dataset_name = 'beauty'
    version_id = '_hao-2020-7-7' # 进程池数量 


    ## 判断output_dir
    if not os.path.isdir(output_dir):
        print('路径不存在::'+str(output_dir))
        exit(1)
    
    dataset = data_partition(output_dir+dataset_name+'.txt')
    user_train, user_valid, user_test, user_num, item_num = dataset

    # user-item pairs, cc 统计一共有多少用户-商品匹配对
    cc = 0
    max_len = 0
    min_len = float('inf')
    for i in user_train:
        cc += len(user_train[i])
        max_len = max(max_len, len(user_train[i]))
        min_len = min(min_len, len(user_train[i]))
    
    # 打印信息
    print('average sequence length: %.2f' % (cc / len(user_train)))
    print('max:{}, min:{}'.format(max_len, min_len))
    print('len_train:{}, len_valid:{}, len_test:{}, user_name:{}, item_num:{}'.format(
        len(user_train), len(user_valid), len(user_test), user_num, item_num))
    
    for idx, u in enumerate(user_train):
        if idx < 10:
            print('user_trian:{} \n user_valid:{} \n user_test:{}'.format(
                  user_train[u], user_valid[u], user_test[u]))
            print('##'*30)
    

    # 训练集加入验证集
    for u in user_train:
        if u in user_valid:
            user_train[u] += user_valid[u]

    # train 和 test 数据格式  {'user_1':[item_2, item_3, ...]}
    user_train_data = {
        'user_'+str(k):['item_'+str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    
    # 测试集为 item_known --> item_pred
    user_test_data = {
        'user_'+str(u):['item_'+str(item) for item in (user_train[u]+user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }


    vocab = FreqVocab(user_test_data)
    user_test_data_output = {k:[vocab.tokens2ids(v)] for k, v in user_test_data.items()}
    # print(user_test_data_output)
    
    
    # 生成训练数据
    print('begin to generate train data')
    output_fname = output_dir+dataset_name+version_id+'.train.tfrecord'
    gen_samples(data=user_test_data,
                output_fname=output_fname,
                rng=rng,
                vocab=vocab,
                max_seq_length=max_seq_length,
                dupe_factor=dupe_factor,
                short_seq_prob=short_seq_prob,
                mask_prob=mask_prob,
                masked_lm_prob=masked_lm_prob,
                max_predictions_per_seq=max_predictions_per_seq,
                prop_sliding_window=prop_sliding_window,
                pool_size=pool_size,
                force_last=False) # False
    print('train:{}'.format(output_fname))


    # 生成测试数据
    print('begin to generate test')
    output_fname = output_dir + dataset_name + version_id + '.test.tfrecord'
    gen_samples(data=user_test_data,
                output_fname=output_fname,
                rng=rng,
                vocab=vocab,
                max_seq_length=max_seq_length,
                dupe_factor=dupe_factor,
                short_seq_prob=short_seq_prob,
                mask_prob=mask_prob,
                masked_lm_prob=masked_lm_prob,
                max_predictions_per_seq=max_predictions_per_seq,
                prop_sliding_window=-1.0,
                pool_size=pool_size,
                force_last=True) # force_last 才是预测
    print('test:{}'.format(output_fname))


    ## 创建 vocab 词表
    print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.format(
            vocab.get_vocab_size(),
            vocab.get_user_count(),
            vocab.get_items_count(),
            vocab.get_items_count() + vocab.get_special_token_count()))


    # 词表保存位置(pickle 文件)
    vocab_file_name = output_dir + dataset_name + version_id + '.vocab'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)


    # 保存user_test_output 文件, 格式为 {'user_id':[给予点击频率的商品ids]}
    his_file_name = output_dir + dataset_name + version_id + '.his'
    print('test data pickle file: ' + his_file_name)
    with open(his_file_name, 'wb') as output_file:
        pickle.dump(user_test_data_output, output_file, protocol=2)
    

    # 完成
    print('DONE!!!')



#%%

if __name__ == "__main__":
    main()
