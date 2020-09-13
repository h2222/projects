# coding=utf-8


import sys, os
import time
import numpy as np
import random
import tensorflow as tf
# 双向 rnn
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# 单个LSTM cell, 实现多层LSTM
from tensorflow.contrib.rnn import BasicLSTMCell


from utils.data_helper import get_data, fill_feed_dict
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 保证生成随机数相同
seed_num = 666



start = lambda : int(round(time.time()*1000))
tolta = lambda x: int(round(time.time()*1000)) - x


class ABLSTM():

    def __init__(self):
        self.max_len = 32
        self.hidden_size = 128 #有128个lstm cell
        self.vocab = './utils/tool/vocab.txt'
        self.embedding_size = 128
        self.n_class = 2
        self.learning_rate = 0.012
        self.cls_type = ['declutter'] # 多任务训练
        self.vocab_size = 1200
        self.layers = 2

        # placehoder
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.string, [None, self.max_len], name='input_x') # 训练sentence [b, max_len]
            self.seqlen = tf.placeholder(tf.int32, [None], name='input_seqlen') # seqlen int
            self.label = tf.placeholder(tf.int32, [None], name='input_y') # lable int
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob') #



    def lstm_cell_with_dropout(self,
                               state_size,
                               keep_prob):

        # RNN (LSTM) 的cell 数目等于输入的句子的长度(时间序列)
        # 每一个 RNN cell 内部, 是有nn神经网络构成的, 神经网络层数 为 LSTM_layer
        

        # 128 word embeding size(input) --> LSTM hidden size(output), LSTM单元中的神经元数量，即输出神经元数量
        # 本例中输入的embedding size 与 输出 hidden_size 相同, 所以 输入 128 输出也128
        cell = tf.contrib.rnn.BasicLSTMCell(num_units = state_size) 

        
        # cell 加 dropout 
        # https://blog.csdn.net/abclhq2005/article/details/78683656
        # LSTM中进行dropout时，对于LSTM的部分不进行dropout，也就是说从t-1时候的状态传递到t时刻进行计算时，
        # 这个中间不进行memory的dropout；仅在同一个t时刻中，多层cell之间传递信息的时候进行dropout
        cell = tf.contrib.rnn.DropoutWrapper(cell=cell,
                                             output_keep_prob = keep_prob,
                                             state_keep_prob = keep_prob,
                                             variational_recurrent=True,
                                             dtype=tf.float32)

        return cell

    def blstm_layer_with_dropout(self,
                                 rnn_inputs,
                                 seqlen,
                                 state_size,
                                 keep_prob,
                                 scope):
        # 前向cell1, 后向cell2
        cell1 = self.lstm_cell_with_dropout(state_size, keep_prob)
        cell2 = self.lstm_cell_with_dropout(state_size, keep_prob)
        
        # output_fw [b, seq_len, hidden_size]   在本实例中 hidden_size 为 128  
        # output_bw [b, seq_len, hidden_size]
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = cell1,
                                                                    cell_bw = cell2,
                                                                    inputs = rnn_inputs,
                                                                    sequence_length = seqlen,
                                                                    dtype = tf.float32,
                                                                    scope = scope) # scope 2 layer LSTM, name_scope
        
        # 前后拼接: [b, seq_len, hidden_size*2=256] 
        return tf.concat([output_fw, output_bw], axis=-1)


    def build_graph(self):
        print('building graph')

        # 建立 embedding table, 将embedding table 嵌入 graph中
        table = tf.contrib.lookup.index_table_from_file(vocabulary_file=self.vocab, num_oov_buckets=1)


        # x为一句话, 根据x中的词, 从vocab中拿到对应的idx, 最大长度32, x不够32补0        
        self.feature_idx = table.lookup(self.x)
        

        # word embedding
        embedding_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), trainable=True)

        # rnn_input:[b, seq_l, emb_sz]
        rnn_input = tf.nn.embedding_lookup(embedding_var, self.feature_idx)


        for i in range(self.layers):
            with tf.name_scope('BLSTM-{}'.format(i)) as scope:
                # runn_input(direction) [b, seq_l, emb_sz * 2]  forward + backword = 128 * 2
                rnn_input = self.blstm_layer_with_dropout(rnn_input,
                                                          self.seqlen,
                                                          self.hidden_size,
                                                          self.keep_prob,
                                                          scope)
        
        # attention
        H = rnn_input
        # Memory:[b, seq_l, hidden_sz*2]
        M = tf.tanh(H)
        # Menory:[b*seq_l, hidden_sz*2] where hidden_sz == emb
        M = tf.reshape(M, [-1, self.hidden_size*2])
        # Weight:[hidden_sz*2, ?]
        W = tf.Variable(tf.random_normal([self.hidden_size * 2], stddev=0.1))
        # Weight:[hidden_sz*2, 1]
        W = tf.reshape(W, [-1, 1])

        # alpha_1 : [?, 1]
        alpha_temp_1 = tf.matmul(M, W)
        alpha_temp_2 = tf.reshape(alpha_temp_1, (-1, self.max_len))
        # alpha_1[b, max_len] # 每一词一个注意力评分
        self.alpha = tf.nn.softmax(alpha_temp_2)
        
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))                

        # [b, hidden_size, 1] -> [b,  hidden_size] 
        r = tf.squeeze(r)
        h_star = tf.tanh(r)
        # attention with dropout
        h_drop = tf.nn.dropout(h_star, self.keep_prob)

        ###  优化器和 全局步(为了rate随着epoch 减少） ###
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)


        ### 多任务训练(每个任务过一个全连接层) ###
        for item in self.cls_type:
            # 命名空间, 参数共享
            with tf.variable_scope(item):
                # 全连接, 权重和偏执 W[输入size: hsz*2, 输入:n_class] 和 b
                FC_W = tf.Variable(tf.truncated_normal([self.hidden_size*2, self.n_class], stddev=0.1))
                FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
                # 全连接输出 y_hat
                y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)


                # 挨个损失加入到当前声明的对象当中(classifier)
                # 名称为 loss_任务名
                # 使用mean-cross_entropy_wthh_logits
                setattr(self, 'loss_'+item,
                        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label)))


                # 任务预测 prediction
                with tf.name_scope('Prediction'):
                    setattr(self, 'prediction_'+item, tf.argmax(tf.nn.softmax(y_hat), 1, name='prediction'))
                

                # 获取loss
                loss_to_minimize = getattr(self, 'loss_'+item) # loss
                tvars = tf.trainable_variables() #  一个list 包含了所有的可训练参数
                gradients = tf.gradients(loss_to_minimize, # 全部梯度 即 DL
                                         tvars, # 全部 可训练参数 即 Dw
                                         aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)  # 减少显存占用 
                    

                # 缩小loss, 防止梯度消失或梯度爆炸, https://blog.csdn.net/u013713117/article/details/56281715                
                grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)


                # 使用优化器进行根据梯度对权重进行更新
                # global_step 最自我增加, 上面定义了global_step 为初始为1不训练的变量
                setattr(self, 'train_'+item+'_op',
                        self.optimizer.apply_gradients(zip(grads, tvars), 
                                                       global_step=self.global_step, 
                                                       name='train_step'))

        print('Graph Building Successfully!!')



def run_train_step(model, sess, batch, class_name, dropout_keep_prob):
    pass
    # feed_dict = make 







if __name__ == "__main__":
    
    train_x, train_y = get_data()


    print(len(train_x))
    print(train_y)


    classifier = ABLSTM()
    classifier.build_graph()

    sess = tf.Session()
    # 初始化全局参数, 初始化embedding_table
    # 初始化embedding_table
    # 测试用: tf.enable_eager_execution()
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    print('n_class:{}\tvocab_size:{}'.format(classifier.n_class, classifier.vocab_size))

    
    ## 开始训练
    the_best_f = 0
    checkpoint_num = 4
    saver = tf.train.Saver(max_to_keep=checkpoint_num)
    train_epoches = 10
    batch_sz = 70

    st = start()
    for epoch in range(train_epoches):
        et = start()
        # print('\n### Current Train Epoch:{} ###'.format(epoch+1))
        for x_batch, y_batch, name_batch, l_batch in fill_feed_dict(train_x, train_y, class_name=classifier.cls_type[0], batch_size=batch_sz):
            print(x)
            print(l_batch)
            # step, loss, accuracy = run_train_step(classifier, sess, (x_batch, y_batch), class_name, dropout_keep_prob)
            


