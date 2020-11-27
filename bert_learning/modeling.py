# coding = utf-8

''' rebuild the bert model'''

from __future__ import print_function # 保证python2.7 能用python3.x的print, 除法, 和 绝对引用
from __future__ import division
from __future__ import absolute_import

import os
import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

# 字符串转ids
from utils import convert_by_vocab

# 设置 gpu/cpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# debug用
good = lambda : print('good')

# 初始化用  (截取部分正太分布并随机采样)
create_initializer = lambda x : tf.truncated_normal_initializer(stddev=x)

# 激活函数 glue
glue = lambda it : it * 0.5 * (1.0 + tf.erf(it / tf.sqrt(2.0)))

# dropout
# it == input_tensor, db == dropout_prob
dropout = lambda it, db: it if db is None or db == 0.0 else tf.nn.dropout(it, 1.0 - db)

# layer_norm 正则化
# it == input_tensor
layer_norm = lambda it, name=None: tf.contrib.layers.layer_norm(it, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

# layer_norm + dropout
def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
    output_tensor = layer_norm(input_tensor, name)
    return dropout(output_tensor, dropout_prob)


# 将 3D tensor 转为 2D matrix
def reshape_to_matrix(input_tensor):
    ndims = input_tensor.shape.ndims

    if ndims < 2:
        raise ValueError('输入的tensor 维度小于2, 转不了2d matrix')
    elif ndims == 2:
        return input_tensor

    # input_tensor的最高维度
    width = input_tensor.shape[-1]
    # 将最高维度之前的维度合并为1个维度
    return tf.reshape(input_tensor, [-1, width])
   
        
# 将 tensor以特定shape的matrix 转为3D tensor
def reshape_from_matrix(output_tensor, orgin_shape_list = None):
    if len(orgin_shape_list) == 2:
        return output_tensor
    # 前几维度为origin_shape_list 中的， 最后一维为out_tensor的 hidden_size
    
    origin_shape = orgin_shape_list[0:-1]
    width = get_shape_list(output_tensor)[-1]
    return tf.reshape(output_tensor, origin_shape + [width])


# 判断一个tensor的rank是否是 期望的rank
def assert_rank(tensor, expected_rank, name=None):

    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    
    # 判断 expect_rank是一个list还是 int
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
        
    # 真实rank , 就是tensor的维度数
    actual_rank = tensor.shape.ndims
    # print('tensor_rank_dict:', expected_rank_dict)
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError('参数rank有问题')
    


# 获取一个tensor的shape数值的list
def get_shape_list(tensor, expected_rank=None, name=None):
    
    if name is None:
        name = tensor.name
    
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    
    shape = tensor.shape.as_list()

    # 用于收集shape中, 维度为None的索引
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    
    # 如果 tensor中没有动态维度, 直接返回shape
    if not non_static_indexes:
        return shape

    # 如果有动态维度, 所有动态维度被替换为 tf.tensor scaler
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


'''
bert 基本参数配置
'''
class BertConfig:

    def __init__(self,
                 vocab_size,  # 词表大小, embedding matrix 的 高
                 hidden_size=786,   # encoder 外部 预测时 使用的神经网络中的隐藏层单元数
                 num_hidden_layers=12, # transformer encoder 中的 feed-forward的网络的隐藏层层数
                 num_attention_heads=12,  # self-attention的多头注意的头的数量
                 intermediate_size=3072,  # transformer encoder 中 
                 hidden_act='gelu',  # 外部神经网络的激活函数
                 hidden_droput_prob=0.1, # 隐藏层的dropout
                 attention_probs_dropout_prob=0.1, # 注意力部分的dropout
                 max_position_embeddings=512, # 最大位置编码. 用于编码词汇的位置信息
                 type_vocab_size=16, # 用于 predict next sentence 任务, 最大可预测16个句子的顺序
                 initializer_range=0.02): #生成随机值的标准差参数, 从截断的正态分布中生成随机值 
                 # tf.truncated_normal_initializer 从截断的正态分布中输出随机值。


        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_droput_prob = hidden_droput_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


    @classmethod
    def from_dict(cls, json_obj):
        '''从字典导入参数 '''
        config = cls(vocab_size=None)
        # py2 dict.items返回的是数组，six.iteritems(dict)则返回生成器。
        # 意味着，dict很大的时候，后者不占用内存。
        # python3 里面， dict.items改变了默认实现，也返回生成器，
        # 因此six.iteritems退出了舞台。
        for (key, value) in six.iteritems(json_obj):
            config.__dict__[key] = value  # __dict__ 函数可以将 实例化对象的属性和值以键值对的形式组成字典
        return config

    @classmethod
    def from_json_file(cls, json_file):
        ''' 从json文件导入参数'''
        # gfile.GFile 类似于 python 中的 open(), 参数为文件名路径, 和操作指令
        with tf.gfile.GFile(json_file, 'r') as reader:
            text = reader.read()
            return cls.from_dict(json.loads(text)) # 奇妙利用了已有的类对象函数, json load 将 text 文件转化为 json

    def to_dict(self):
        '''将配置参数以字典形式导出 '''
        return copy.deepcopy(self.__dict__)
        
    
    def to_json_string(self):
        ''' 将json对象转为string 并 修改格式, 排序顺序 '''
        # indent 代表json缩进, sort_keys为true代表按照keys的大小排序
        return json.dumps(self.to_dict(), indent=4, sort_keys=True) + "\n"




''' 输入 word 的 ids (通过统计所以词的数量统计得来), 获取词向量table和词向量,
 将每词根据他的词表排序(ids) 根据 embedding_table 转化为一个128 维度的词向量, embedding table 是可训练参数

tf func:
tf.nn.embedding_lookup(params, ids) tf.cast(attention_mask, tf.float32
    params 是一个 可以是一个tensor
    ids 也可以是一个 tensor

    例如 embedding_table 的 shape 为 (1000, 128) 表示1000个词, 每个词是128d
    的 embedding vector

    ids 为 one_hot_matrix , shape为(1, 7, 1)
    
    最后 查询结果为的shape为 (1, 7, 1, 128) 
    可以解释为为第一个batch的7个tokens的第1个元素的在embedding_table
    的embedding_vector(128)
 '''
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size = 128,
                     initializer_range=0.02,
                     word_embedding_name='word_embedding',
                     use_one_hot_embedding=True):
    
    print("**embedding_lookup**"*5)
    # 该函数默认输入的形状为[batch_size, seq_length, input_num]
    # 如果输入为2D的[batch_size, seq_length]，则扩展到[batch_size, seq_length, 1]
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])


    # 映射表 (1000, 128)
    # tf.Variable 和 tf.get_variable 
    # tf.Varianle 可以不用变了名, 且创建多个变了变量名可以一样(系统会自动帮你修改), 可使用 v.name查看
    # tf.get_variable 在同一个命名空间下必须设置为不同的名称, 因为get_variable支持向量共享
    # https://blog.csdn.net/TeFuirnever/article/details/89577480?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task
    embedding_table = tf.get_variable(name=word_embedding_name, 
                                      shape=[vocab_size, embedding_size],
                                      initializer=create_initializer(initializer_range))

    
    # 使用 one-hot 编码
    if use_one_hot_embedding:              # 最外层    中间层    内层
        # reshape -1 为打平, 原 shape 为 [batch_sz, input_ids, input_nums]
        # 打平为 [batch_sz * input_ids * input_nums]
        flat_input_ids = tf.reshape(input_ids, [-1])
        # one-hot, depth参数, one-hot长度
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        # 通过 每个词的one-hot编码, 在embedding matrix找到对应的 embedding
        output = tf.matmul(one_hot_input_ids, embedding_table)
    else:
        #  embedding_table 为一个list或者tensor
        # input_ids 为一个tensor
        # 在 embedding_table 中 查找input_ids的对应位置
        output = tf.nn.embedding_lookup(embedding_table, input_ids)

    input_shape = get_shape_list(input_ids)

    print('input_shape(embedding step):', input_shape)
    print('output(not reshape, embedding step):', output.shape)
    print('embedding_table shape (embedding step)', embedding_table.shape)

    # 检测用, 当输入进的 input_ids batchsz为1, 每个batch的ids为7时,
    # input_ids 的shape为 (1,7, 1) 因为扩增了1
    # [0:-1] -1 取不到所以为 最后一维 1
    # embeding_sz 为 128 
    # 所以outputs 的shape为 (1, 7, 128)
    # print(input_shape[0:-1])
    # print([input_shape[-1]* embedding_size])
    # print(input_shape[0:-1] + [input_shape[-1]* embedding_size])
    
    # output shape -> [最大] (该步骤的目的为还原batch)
    output = tf.reshape(output,
                        input_shape[0:-1] + [input_shape[-1] * embedding_size])
    
    return (output, embedding_table)


'''
词向量的后续处理
bert输入模型有三部分: token embedding, segement embedding position embedding
在 embedding_lookup 中我们只获得 token_embedding 即(outputs)

该部分代码, 对其完整信息, 正则化, dorpout后, 最终输出的embedding送入self-attention 层

注意，在Transformer论文中的position embedding是由sin/cos函数生成的固定的值，
而在这里代码实现中是跟普通word embedding一样随机生成的，可以训练的。
作者这里这样选择的原因可能是BERT训练的数据比Transformer那篇大很多，
完全可以让模型自己去学习。
https://blog.csdn.net/Kaiyuan_sjtu/article/details/90265473

tf func:
tf.get_varaible(name, shape, initializer) name 必须有
tf.one_hot(indices=a_tensor, depth=int, axis=int) one-hot tensor
    axis 1表示按列排列one_hot, 0表示按行排列
    depth 表示几个数需要one_hot
tf.assert_less_equal(num1, num2) 比大小
with tf.control_dependencies([assert_op]):  控制依赖, 只有xx成立才能继续续往下
tf.silce(input, begin, size, name)  begin[1, 2, 3, 4] 每个维度的启示元素, size每个维度切多少
'''
def embedding_postprocessor(input_tensor,  # # [batch_size, seq_length, embedding_size] 与 embedding_lookup的输出相同
                            use_token_type=False,
                            token_type_ids=None,  # int32 Tensor of shape [batch_size, seq_length].Must be specified if `use_token_type` is True.
                            token_type_vocab_size=16, # the vocabulary size of 'token_type_ids', nomally 16
                            token_type_embedding_name='token_type_embeddings',  # 就是 segement , 第一句话就是 0, 第二句话就是1
                            use_position_embedding=True,
                            position_embedding_name='position_embeddings',
                            initializer_range=0.02,
                            max_position_embeddings=512, # 最大 位置编码， 必须和输入的词(input_ids 的 seq_length)一样多
                            dorpout_prob=0.1):

    print('***embeddding_postprocessor***'*5)
    # 因为 input_tensor的shape为 [batch_sz, seq_length, emebedding_size]
    # 所有 秩 rank 为 3
    print('input_tensor shape:', input_tensor.shape)
    input_shape =  get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2] # embedding_size  # segement_ids 的长度必须与句长相同 

    # 初始化output为input, 主要是为了后面相加的时候shape相同
    # shape 为 (1, 7, 128)
    output = input_tensor


    if use_token_type:
        if token_type_ids is None:
            raise ValueError('token type ids 必须有')
            
        # token type 指的就是 bert 的 next sentence 任务， 
        # token_type_table 的 shape 为 (16,128)  16 为token_type_ids的大小， 
        # 每句话 128的 segment embedding
        token_type_table = tf.get_variable(name=token_type_embedding_name,
                                            shape=[token_type_vocab_size, width],
                                            initializer=create_initializer(initializer_range))
        print('token_type_table 的shape, 用于与one_hot_type_ids改造为与output_tensor的shape1相同的tensor, 方便叠加:', token_type_table.shape)
        # 由于token-type-table 比较小, 所有这里采用one-hot的embedding的方式加速
        # token_type_ids 为你在 embedding_lookup 输入的东西
        print('token_type_ids的初始shape:', token_type_ids.shape)
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        # 打平后 为  (7,)
        print('segment的type的类别:',flat_token_type_ids.shape)

        # depth 表示使用one_hot 向量最大的向量长度
        # one_hot_ids (7, 16)
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        print('使用one_hot 的embedding', one_hot_ids.shape)

        # 使用one-hot ids 乘以 token_type_table
        # token_type_embedding 的 shape为 (7, 16) * (16, 128) --> (7, 128)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        print('token_type_embedding:',token_type_embeddings.shape)

        # 将shape修改为与output 相同,即 (1, 7, 128)
        token_type_embeddings = tf.reshape(token_type_embeddings, [batch_size, seq_length, width])
        print('修改shape后的token_type_embedding:', token_type_embeddings.shape)

        # 加入token_type_embedding到 output 中, 为各维度上的数值相加, 即算是嵌入了token_embedding, (token_type_embedding 就是 segment_embedding)
        output += token_type_embeddings
        print(output.shape)


    ## position embeeding ## 位置信息嵌入
    if use_position_embedding:

        # 确保 seq_length <= max_position_embedding
        assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):

            # full_position_embedding shape (512, 128)  512 是最大sequence length
            full_position_embeddings = tf.get_variable(name=position_embedding_name,
                                                       shape=[max_position_embeddings, width],
                                                       initializer=create_initializer(initializer_range))
            print('full_postition_embedding_table, 用于给output_tesnor加positional_embedding:', full_position_embeddings.shape)            
            # 这里position embedding是可学习的参数，[max_position_embeddings, width]
            # 但是通常实际输入序列没有达到max_position_embeddings
            # 所以为了提高训练速度，使用tf.slice取出句子长度的embedding
            # tf.slice(input, begin, size, name=None)
            # 举例 现在 t = [[[1,1,1],[2,2,2]] [[3,3,3],[4,4,4]], [[5,5,5],[6,6,6]]]
            # t 的 shape 为 (3, 2, 3)  一共三个维度
            # 1维  t = [A, B, C]
            # 2维  A = [a, b]  B = [c, d]  C = [e, f]
            # 3维  a = [1, 1, 1]  b = [2, 2, 2] c = [3, 3, 3] d = [4, 4, 4] e = [5, 5, 5] f = [6, 6, 6]
            # beign 只的就是从某一维度的某一个值开始算, 例如begin=(1, 1, 0)的含义就是
            # 取1维度的B, 在2维度中取B中的c, 在三维度中取c中的3

            # size 参数 代表在某一个维度取值的个数, 例如size = (1, 1, 3) 时
            # 1维度只取B因为1, 2维度只取c因为1, 3维度取[3, 3, 3]因为3

            # 根据slice 的解释, full_position_embedding的shape为(512, 128)
            # 从1维度从第0个开始取seq_length个元素, 2维度下从0开始全部取出
            # position_embedding 的shape 为 (7, 128) 
            position_embeddings = tf.slice(full_position_embeddings, 
                                            [0, 0],
                                            [seq_length, -1])
            
            print('postion_embedding 的 shape(没对齐之前):',position_embeddings.shape)
            # nums_dims = [1, 7, 128]
            # len = 3
            num_dims = len(output.shape.as_list())

            # word embedding之后的tensor是[batch_size, seq_length, width]
            # 因为位置编码是与输入内容无关，它的shape总是[seq_length, width]
            # 我们无法把位置Embedding加到word embedding上
            # 因此我们需要扩展位置编码为[1, seq_length, width]
            # 然后就能通过broadcasting加上去了。
            position_broadcast_shape = []
            # 循环一次, 3-2 = 1次, 只增加1维度
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            
            # 维度增加
            position_broadcast_shape += [seq_length, width]
            # shape 统一为 [1, seq_length, embedding_size]
            position_embeddings = tf.reshape(position_embeddings,
                                             position_broadcast_shape)

            print('postion_embedding 的 shape:',position_embeddings.shape)

            # position_embedding 嵌入到 output, 各维度数据相加
            output += position_embeddings

            # embedding 层的 正则化 与 dropout
            output = layer_norm_and_dropout(output, dorpout_prob)
            print('output的shape:',output.shape)

            return output    


'''
创造 attention_mask (能让模型做完形填空的关键)
tf func:
tf.one(shape=[...], dtype=tf.float32) 全1矩阵
tf.reshape
'''
def create_attention_mask_from_input_mask(input_ids, input_mask):
    print('***create_input_mask***'*7)
    # 输入的 inputs_ids 的shape必须为[batch_size, seq_length, embedding_size] shape=3
    # 或者 shape 为 [batch_size, seq_length] shape = 2
    from_shape = get_shape_list(input_ids, expected_rank=[2, 3])
    
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    print('mask 步骤, 输入seq_length:',from_seq_length)

    # 输入的 mask_ids shape 必须为 [batch_size, to_seq_length]
    # 做attention_mask的原因是希望bert模型做完形填空, 预测mask部分的词
    # 所以mask的总总长度必须与 seq_length 相同, 只需要2维度因为 mask 不需要embedding
    to_shape = get_shape_list(input_mask, expected_rank=2)
    to_seq_length = to_shape[1]
    print('mask 总长度:', to_seq_length)

    # to_mask shape -> [batch_size, 1, seq_length]
    # 本次实验输入的 imput_mask 为 [[1,1,1,1,1,1,1]] 加1维度 为 [[[1,1,1,1,1,1,1]]]
    input_mask = tf.cast(tf.reshape(input_mask, [batch_size, 1, to_seq_length]), tf.float32)
    print('input_mask变换维度后shape:', input_mask.shape)
    # broadcast_ones -> [batch_size, seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, from_seq_length, 1], dtype=tf.float32)    
    print('中间变量one矩阵:', broadcast_ones.shape)
    # MASK (1, 7, 1) * (1, 1, 7) --> (1, 7, 7)
    # attention_mask 
    attention_mask = broadcast_ones * input_mask
    print('最终mask结果', attention_mask.shape)        
    return attention_mask

'''
这部分代码是multi-head attention的实现，主要来自《Attention is all you need》这篇论文。
考虑key-query-value形式的attention，输入的from_tensor当做是query， to_tensor当做是key和value，
当两者相同的时候即为self-attention。

tf func:
tf.reshape(tensor, [ dim list])
tf.transpose(tensor, [dim order list])
tf.layers.dence(input_tensor, units, ...)  全连接层建立
tf.matmul(tensor1, tensor2, transpose_b=True) 矩阵乘法
tf.multiply(number1, number2) 数值相乘
tf.expand_dims(tensor, axis=[1]) 在 axis所在的维度增加1维度
tf.cast(attention_mask, tf.float32) 将 tensor中的元素转为某个特定类型
tf.nn.softmax(tensor) softmax 函数

'''
def attention (from_tensor, # [batch_size, from_seq_length, from_width], 这里的 from_tensor即为经过embedding_lookup 和 postpreprocessing的 token vector
               to_tensor, # [batch_size, to_seq_length, to_width]
               attention_mask=None, # [batch_size, from_seq_length, to_seq_length]
               num_attention_heads=1, # attention head 数目
               size_per_head=521, # 每个head的大小, 即每个head每次都能主要最大seq_length个词
               query_act=None,# query 的激活函数, query 即为当前token
               key_act=None,# key 的激活函数, key向量不仅当前token的query向量, 也需要跟其他token进行处理
               value_act=None, # value的激活函数, value向量为经过一些q 和 k 的 score 处理后, 与每一个value相乘
               attention_probs_dropout_prob=0.0, # attention层的dropout
               initializer_range=0.02, # 初始化范围
               do_return_2d_tensor=True, # 是否返回 2d tensor
               # 如果 true, 返回 [batch_size * from_seq_length, num_attention_heads(1) * size_per_head(512)]
               # 如果 false, 返回 [batch_size, from_seq_length, num_attention_heads*size_pre_head]
               batch_size=None, # 如果输入 3d tensor, 那么batch就是1维度, 但是可能3d的压缩成2d的, 所以需要告诉函数batch_size
               from_seq_length=None,# 同上
               to_seq_length=None): # 同上


    print('***attention***'*8)
    # 闭包函数 (为打分而转制)
    def trnaspose_for_score(input_tensor, batch_size, num_attention_heads, seq_length, width):
        
        # 转成多头：[B*F, N*H] ==>  [B, F, N, H]  ==> [B, N, F, H]
        #         input_tensor       output_tensor       return

        #       0            1           2                3
        # [batch_size, seq_length, num_attention_heads, width]
        output_tensor = tf.reshape(input_tensor, 
                                   [batch_size, seq_length, num_attention_heads, width])

        #      0             2                  1         3
        # [batch_size, num_attention_heads, seq_length, width]
        return tf.transpose(output_tensor, [0, 2, 1, 3])


    # input_tensor, output_tensor rank 检查, 并返回对应shape 的 list
    from_shape = get_shape_list(from_tensor, expected_rank=[2,  3]) # 至少2维度最多3维度
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])


    if len(from_shape) != len(to_shape):
        raise ValueError("你这input和output维度对不上呀")

    
    # 输入 2d tensor 或者 3d tensor
    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = from_shape[1] # 保证 input_seq和output_seq的长度一样
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError("你这batch都没size的呀, 或者你输入的tensor里面维度有几个是空值, 找不大seq_length呀")


    # 为了方便, 下面使用简写
    #   B = batch_size(一个批次里面有多少sequence)
    #   F = from_tensor 的当中一个sequence的长度
    #   T = to_tensor输入tensor的一个sequence的长度, 与输入的一样
    #   N = num_attention_heads 头数
    #   H = size_pre_head 每个头的最大注意数 512


    # 把 from_tensor(3d) 和 to_tensor(3d) 压缩为2d tensor
    from_tensor_2d = reshape_to_matrix(from_tensor)  # [B*F, hidden_size] 这里的hidden_size即等于token_embedding的1size即 128
    to_tensor_2d = reshape_to_matrix(to_tensor) #  [B*T, hidden_size] 

    # [B*F, hidden_size] == (7, 128)
    print('2d化的input_tensor', from_tensor_2d.shape)


    # 将from_tensor 输入到全连接层得到query_layer
    # 
    # query_layer 的 shape 为 [B*F, N*H]
    # tf.layers.dence 全连接层 densely connected layer 
    # 主要参数 
    # inputs: tensor input
    # units: 一个 int or long, 表示output tensor 的维度, 即全连接层的hidden_size
    # 全连接层, 只有输入层和输入层, 没有隐藏层
    query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads*size_per_head,
        activation=query_act,
        name='query',
        kernel_initializer=create_initializer(initializer_range)
    ) 

    # query shape : [BF, NF] == (1*7, 1*512 )
    print('query shape:', query_layer.shape)

    # 将to_tensor输入到全连接层得到key_layer
    # key_layer [B*T, N*H]
    key_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=key_act,
        name='key',
        kernel_initializer=create_initializer(initializer_range)
    )


    # 将 to_tensor 输入到全连接层得到value_layer
    # value_tensor = [B*T, N*H]
    value_layer = tf.layers.dense(
        to_tensor_2d,
        num_attention_heads * size_per_head,
        activation=value_act,
        name='value',
        kernel_initializer=create_initializer(initializer_range)
    )

    # query_layer 转成多头模型的tensor shape:
    # [B*F, N*H] ==> [B, F, N, H] ==> [B, N, F, H]
    query_layer = trnaspose_for_score(query_layer, 
                                      batch_size, # B
                                      num_attention_heads, # N
                                      from_seq_length, # F
                                      size_per_head) # H
    
    print('多头化后的 query shape(用于attention计算):', query_layer.shape)
    
    # key_layer 转为多头模型的tensor shape:
    # [B*T, N*H] ==> [B, T, N, H] ==> [B, N, T, H]
    key_layer = trnaspose_for_score(key_layer,
                                    batch_size,
                                    num_attention_heads,
                                    from_seq_length,
                                    size_per_head)
    print('多头化后的 key shape(用于attention计算):', key_layer.shape)
    
    # 定义的value_layer的shape
    # [B*T, N*H] ==> [B, T, N, H] ==> [B, N, T, H]
    # T 对应 to_tensor的sequence_length
    value_layer = trnaspose_for_score(value_layer,
                                      batch_size,
                                      num_attention_heads,
                                      to_seq_length,
                                      size_per_head)
    print('多头化后的 value shape(用于attention计算):', value_layer.shape)

    # 算当前query的attention的分数   (query * key ^ T) / sqrt(size_pre_head)
    # 即 [B, N, F, H] · [B, N, T, H]^T == > [B, N, F, T] (1, 1, 7, 7)
    # 将query和key做点积, 然后做一个scale
    # [B, N, F, T] / sqrt(512) 表示 scale
    # 为什么要加 1/sqrt(H) ? 论文在p4 解释道
    #While for small values of dk the two mechanisms perform similarly, additive attention outperforms
    #dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
    # 1/sqrt(H) , the dot products grow large in magnitude, pushing the softmax function into regions where it has
    #extremely small gradients 4. To counteract this effect, we scale the dot products by 1/sqrt(H) 

    #解释为当 每个attention head 需要注意很少的词的时候, 1/sqrt(H) 加与不加没有什么区别, 但随着 H的增大
    # 将 attention_score的增大会造成 softmax 的 趋近 0 或 1 导致梯度消失, 所以加 1/sqrt(H) 做scale化

    # matmul - matrix multiply   multiply - regualer multiply
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 
                                   1.0 / math.sqrt(float(size_per_head)))

    print('(Q · K^T)/sqrt(H) 后的shape:', attention_scores.shape)


    # 给 attention 上 mask
    if attention_mask is not None:

        # 原来的 attention_mask shape为 [B, F, T]  即(1, 7, 7)      
        # attention_mask 修改 shape 使其匹配 attention_score 的 shape
        # [B, 1, F, T]   N = 1 因为 mask为该toke的attention 被mask, 不需要考虑attention_head
        # (1, 1, 7, 7)
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # 如果attention_mask里面的元素为1, 则通过下面的运算有(1-1)* -10000, adder=0
        # 如果attention_mask里面的元素为0, 则通过下面的运算有(1-0)* -10000, adder = -10000
        adder = (1.0 - tf.cast(attention_mask, tf.float32))* -10000.0

        # 如果 adder 为 0 了对attention没啥改变, 如果 adder为 -10000 了
        # 得到的attention_score 就非常非常小, 基本上为负无穷
        # += 在 tensor之间为同纬度元素相加, 且需要维度相同
        # 所以 adder 对所有query都有效果
        attention_scores += adder

        # 如果 attention_score 为 负无穷, 过softmax就肯定为0了 [B, N, F, T](1,1,7,7)
        attention_probs = tf.nn.softmax(attention_scores)


        # 对attention_prob进行dropout, 虽然有点奇怪, 但论文就在这么做得
        attention_probs = dropout(attention_probs, attention_probs_dropout_prob)


        # attention(Q, K, V) == softmax(Q · K^T/sqrt(H)) · V
        # [B, N, F, T] * [B, N, T, H] ==> [B, N, F, H] (1, 1, 7, 512)
        context_layer = tf.matmul(attention_probs, value_layer)

        # 转换context 的shape 到 [B, F, N, H] (1, 7, 1, 512)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        if do_return_2d_tensor:
            # [B, F, N, H] ==> [B*F, N*H] (7, 512)
            context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head]
            )

        else:
            # 返回 3d tensor [B, F, N*H] (1, 7, 512)
            context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head]
            )

        print('attention 返回结果', context_layer.shape)
        return context_layer


'''
Transformer 
"attention is all you need"

tf func:
tf.variable_scope("name", reuse=True) 参数共享
tf.concat(tensor1, tensor2, axis=1)
'''


def transformer_model(input_tensor,  # 经过embedding 和 postprocessing 处理的 tensor [batch_size, seq_length, hidden_size] (1, 7, 128) hidden_size 与 embedding_size对应
                      attention_mask=None,  # attention_mask
                      hidden_size=128, # 768
                      num_hidden_layers=12,
                      num_attention_heads=8, # 129 // 16 == 8
                      intermediate_size=3072,
                      intermediate_act_fn=None, # encoder 中 feed-forward 的激活函数
                      hidden_droput_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.2,
                      do_return_all_layers=False): 

    # 注意, 因为最终输入的 hidden_size, 因为 attention为的multi_head_attention
    # 所以每个 head区域有size_per_head个隐层
    # 所以 hidden_size = num_attention_heads * size_per_head  (1 * 512)
    # 详情参考 https://jalammar.github.io/illustrated-transformer/ multi_head 部分
    print('***transformer_model***'*7)


    if hidden_size % num_attention_heads != 0:
        raise ValueError("多头注意没有头, 隐层的size不能被头整除")

    
    # attention_head_size (一个头的)  16
    attention_head_size = int(hidden_size / num_attention_heads)
    print('每个头的大小', attention_head_size)

    # 获取 input_tensor维度参数 [batch_sz, seq_length, hidden_size]
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]


    # 因为 encoder 中有残差操作, 所以需要 input_hidden_size 等于 output_hidden_size
    if input_width != hidden_size:
        raise ValueError("输入的hidden_size(embedding_size) 与 输入hidden_size大小不同, 不能进行残差计算")


    # reshape 操作在CPU/GPU上很快, 但在TPU上不是很友好
    # 所以应该避免2d和3d的频繁reshape, 我们把所以3d tensor使用 2d 表示
    prev_output = reshape_to_matrix(input_tensor)
    print('attention的输入tensor:', prev_output.shape)
    print('attention_mask:', attention_mask.shape)  # (1, 7, 7)


    #  transformer 12 层 encoder 循换 (每层 hidden_size = 768, 输入 )
    all_layer_outputs = []
    '''
    bert (encoder * 12)
        encoder
            multi head self-attention (self-attention * 8) 
                self-attention (multi head 在 self-attention内以m:atrix的方式实现即, [B*F, N*H] N为heads数量, H为每个head的大小)
                    input_tensor(or prev_output)
                        |
                    attention_layer
                        |
                    one attention head
                densely connected network (8*attention_head_size -> hidden_size)
                dropout+layer_normal+add
                intermediate_dnn (hidden_size -> intermediate_size (3072))
                densely conntect network (intermediat_size -> hidden_size (128))
                dropout+layer_normal+add
        output send to next endcoder
    '''
    for layer_idx in range(num_hidden_layers):
        # 参数共享
        with tf.variable_scope("layer_%d" % layer_idx):
            # 进入 transformer时 [1, 1, 7, 128] --> [1, 1, 7, 768]
            # encoder 与 encoder 输入等于下一层encoder的输入
            layer_input = prev_output
            # multi-head attention
            with tf.variable_scope('multi_head_attention'):
                attention_heads = []
                # self-attention
                with tf.variable_scope('self_attention'):
                    attention_head = attention(
                        from_tensor=layer_input, 
                        to_tensor=layer_input,  # 输入 tensor与输出tensor 相同 (7, 128) 2d tensor
                        attention_mask=attention_mask, # (1, 7, 7)
                        num_attention_heads=num_attention_heads, # 8
                        size_per_head=attention_head_size, # 128 / 8 = 16
                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                        initializer_range=initializer_range,
                        do_return_2d_tensor=True,
                        batch_size=batch_size,
                        from_seq_length=seq_length, # 7
                        to_seq_length=seq_length)   # 7

                    # 多头收集
                    print('单个头的attention 输出 shape:', attention_head.shape)
                    attention_heads.append(attention_head)
                
                # attention_output 初始化
                attention_output = None

                if len(attention_heads) == 1: # 单头注意
                    attention_output = attention_heads[0]
                    print('单头注意力attention_layer后, 送入FNN前, shape为:', attention_output.shape)
                else: 
                    # 多头拼接, 将每个头拼接起来并乘以学习矩阵使其下一层的embedding仍然是128
                    attention_output = tf.concat(attention_heads, axis=-1) # 最后一维度拼接
                    print('出多头attention_layer后, 送入FNN前, 进行多头拼接, shape为:', attention_output.shape)
                    print('最后一个维度大小为 head_num * size_per_head')
                # 对attention的输出进行线性映射, 目的是将shape变成与input_tensor一致,方便送入下一个encoder
                # [B*F, N*H] -> [B*F, hidden_size] (7, 128)
                with  tf.variable_scope('output'):
                    attention_output = tf.layers.dense(
                        attention_output, # input 128 * 8
                        hidden_size, # output 128
                        kernel_initializer=create_initializer(initializer_range)
                    )

                    # encoder 中在attention后的 dropout
                    attention_output = dropout(attention_output, hidden_droput_prob)
                    # encoder中的在attention后的 layer_normalization(维度数值相加)
                    attention_output = layer_norm(attention_output + layer_input)

            # encoder 中的 全连接层 [B*F, hidden_size] -> [B*F, intermediate_size] (7, 3072)
            with tf.variable_scope("intermediate_dnn"):
                intermediate_output = tf.layers.dense(
                    attention_output,
                    intermediate_size,
                    activation=intermediate_act_fn,
                    kernel_initializer=create_initializer(initializer_range)
                )
            

            # 对 dnn 的的输入结果做线性变化使变回'hidden_size' 使每个encoder的输入维度相同 [B*F, intermediate_size ] -> [B*F, hidden_size] (7, 128)
            with tf.variable_scope('output'):
                layer_output = tf.layers.dense(
                    intermediate_output,
                    hidden_size,
                    kernel_initializer=create_initializer(initializer_range)
                )

                # 在加dropout 和 layer_normal
                layer_output = dropout(layer_output, hidden_droput_prob)
                layer_output = layer_norm(layer_output + attention_output)
                
                # layer_output 转为 prev_output, 下一次循送入 self-attention 中
                prev_output = layer_output

                # all_layer_output
                all_layer_outputs.append(layer_output)

    # 如果需要返回每一层encoder的结果
    if do_return_all_layers:   
        final_outputs = []
        for layer_output in all_layer_outputs:
            # 将每层的输出 tensor 维度调整为与 input_tensor 一致 []
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_output
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
    
        # 返回结果
        print('transformer final result', final_output.shape)
        return final_output
            


'''
入口函数

BertModel

tf func:
tf.ones
tf.zeros
tf.squeeze(tensor, axis=1) axis 为需要挤压的维度

tensor 也是可以切片的
tensor1[:, 0:1, :]  --> tensor[b, 1, h]
'''

class BertModel:
    def __init__(self,
                 config,
                 is_training,
                 input_ids, # 词根据词表查的位置 [batch_size, seq_length]
                 input_mask, # 完形填空遮盖 1为不遮盖0为遮盖 [batch_size, seq_length]
                 token_type_ids=None, # 判断下一句任务 token不同的数字代表不同的token属于不同的任务
                 use_one_hot_embeddings=False,
                 scope=None):

        config = copy.deepcopy(config)

        # 如果不是训练, 不进行dropout
        if not is_training:
            config.hidden_droput_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        # 维度数据提取
        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]


        # 不mask, 所有数据mask 为 1:
        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        # 不进行下一句任务预测, 所有token 属于任务0
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        
        with tf.variable_scope(scope, default_name='bert'):
            with tf.variable_scope('embeddings'):
                # word embedding (embedding_table 作为参数进行训练)
                # input_ids 转为 embedding_lookup
                # [B, F] --> [B, F, hidden_size (就是emb_size)]
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name='word_embeddings',
                    use_one_hot_embedding=use_one_hot_embeddings
                )

                # embedding_postprocessing
                # 在 原有 embedding_lookup的基础上加入 position embedding
                # 和 segment embedding (token_type_ids 的 emb)
                # layer_normal + droput
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_embedding_name='token_type_embeddings',
                    use_position_embedding=True,
                    position_embedding_name='position_embeddings', # postion_embedding 作为训练参数是可以被训练的
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dorpout_prob=config.hidden_droput_prob
                )

            # encoder
            with tf.variable_scope("encoder"):
                # input_ids : 一条sequence样例: [25, 120, 34, ..]  size为max_length length = 36
                #  create_mask_and_padding(..., max_lenght=36)
                # input_mask 一条sequence样例: [1, 1, 1, 1, ...] size为 max_length
                attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)


                # transformer模型叠加
                # transformer最后输出结果 sequence_output [batch_size, seq_length, hidden_size]
                self.all_encoder_layers = transformer_model(
                    input_tensor = self.embedding_output,
                    attention_mask=attention_mask,
                    hidden_size=config.hidden_size,
                    num_hidden_layers=config.num_hidden_layers,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    intermediate_act_fn=glue,
                    hidden_droput_prob=config.hidden_droput_prob,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    do_return_all_layers=True
                )

                # sequence_output 是 all_encoder_layers 的最后一层
                self.sequence_output = self.all_encoder_layers


                # pooler 层, [batch_size, seq_length, hidden_size] -- > [batch_size, hidden_size]
                with tf.variable_scope("pooler"):
                    # 取最后一层的 [cls]时刻(即第一个时刻)对应的tensor,用于进行分类服务
                    # sequence_output [:, 0:1, :]得到就是[cls]的tensor [batch_size, 1, hidden_size]
                    # 然后使用 squeeze将 seq_length维度去掉, 方便后面dnn分类
                    first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)

                    # 加一个全连接层, 输入保持 [batch_size, hidden_size]
                    self.pooled_output = tf.layers.dense(
                        first_token_tensor,
                        config.hidden_size,
                        activation=tf.tanh,
                        kernel_initializer=create_initializer(config.initializer_range)
                    )


        # bert model 最后输出(1, 128) 1 为 token [cls], 128 为 embedding_size
        print('bert mode output:', self.pooled_output.shape)
    

    '''bert最终结果 [1, hidden_size]'''
    def get_pooled_output(self):
        return self.pooled_output
    
    '''经过transformer后最后一层encoder输出结果 [batch_size, seq_length, hidden_size]
        final hidden layer of encoder
     '''
    def get_sequence_output(self):
        return self.sequence_output

    '''经过transfermer的所有encoder的输出集合, list [[batch_size, seq_length, hidden_size], [encoder2], [encoder3],...]
        all hidden layer of encoders
     '''
    def get_all_encoder_layers(self):
         return self.get_all_encoder_layers

    ''' input_ids, 经过 embedding， postprocessing 后的输出 [batch_size, seq_length, hidden_size]
        after summing the word embeddings with the positional embeddings and the tokens type embeddings,
        the performing layer normalizatio. this is the input to the transformer 
    '''
    def get_embedding_output(self):
        return self.embedding_output

    '''
    embedding table 可训练, 使用 input_ids 在 embedding table 找到ids对于的embedding
    '''
    def get_embedding_table(self):
        return self.embedding_table

                




# 测试
if __name__ == "__main__":
    # d = {'vocab_size':500}
    # b = BertConfig.from_dict(d)
    # print(b.to_json_string())
    input_ids = convert_by_vocab("大叔大婶大所多")  
    print("input_ids:", input_ids) 
    input_ids = tf.constant([input_ids])
    # segment_embedding, 表示该句子的7个词都属于该句子
    # 如果有两句话则可以表示为 tf.constant([[1,1,...], [2,2,...]])
    #token_type_ids = tf.constant([[1,1,1,1,1,1,1]])
    token_type_ids = tf.constant([[1,1,1,1,1,1,1]])
    # 表示该句子中, 第4个词被mask
    input_mask = tf.constant([[1,1,1,0,1,1,0]])



    # 独立测试
    # with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    # print(input_ids)
    x, y = embedding_lookup(input_ids, 1000)

    # print(x.shape)
    output = embedding_postprocessor(x, use_token_type=True, token_type_ids=token_type_ids)
    print(output.shape.ndims)

    #assert_rank(output, [2, 3])

    # MASK 测试
    attention_mask = create_attention_mask_from_input_mask(output, input_mask)


    # attention test (1 head)
    # attention(output, output, attention_mask=attention_mask)


    # transformer test (12 head)
    # output 为 embeding 处理后的结果
    transformer_model(output, attention_mask, intermediate_act_fn=glue, num_hidden_layers=1)




# 创建BertConfig实例
# config = BertConfig(vocab_size=32000, 
#                     hidden_size=128,
#                     num_hidden_layers=12,
#                     num_attention_heads=8, 
#                     intermediate_size=1024)


# # 创建BertModel实例
# model = BertModel(config=config,
#                   is_training=True,
#                   input_ids=input_ids,
#                   input_mask=input_mask,
#                   token_type_ids=token_type_ids)

# # 10class 分类
# label_embedding = tf.get_variable(name='label_embedding', shape=[config.hidden_size, 10], initializer=create_initializer(0.2))
# # label_embedding = tf.get_variable(...)
# pooled_output = model.get_pooled_output()

# logits = tf.matmul(pooled_output, label_embedding)

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# result = sess.run(tf.nn.softmax(logits, axis=-1))

# # argmax
# result = tf.argmax(result, axis=-1)
# print(sess.run(result)) #  class 3 概率最大








    