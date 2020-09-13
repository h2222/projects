#coding=utf-8
import os
import optimization
import tensorflow as tf
import numpy as np
import sys
import pickle

import modeling


# class EvalHooks(tf.train.SessionRunHook):
    # def __init__(self):
    #     print('run init')
    
    # def begin(self, 
    #           user_history_filename):
    #     self.valid_user = 0.0
    #     self.ndcg_1 = 0.0
    #     self.hit_1 = 0.0
    #     self.ndcg_5 = 0.0
    #     self.hit_5 = 0.0
    #     self.ndcg_10 = 0.0
    #     self.hit_10 = 0.0
    #     self.ap = 0.0
    #     np.ramdom.seed(12345)
    #     self.vocab = None
    
    #     if user_history_filename is None:
    #         print('load user history from:'+user_history_filename)
            
    #         with open(user_history_filename, 'rb') as f:
    #             self.user_history = pickle.load(f)
        
    #     if vocab_filename is not None:
    #         print('load vocab from:'+vocab_filename)
    #         w
    #         with open(vocab_filename, 'rb') as f:
    #             self.vocab = pickle.load(f)
        
    #     keys = self.vocab.counter.keys()
    #     values = self.vocab.counter.values()
    #     self.ids = self.vocab.token2ids(keys)

    #     ## normalize
    #     print(values)
    #     sum_value = np.sum([x for x in values])
    #     print(sum_value)
    #     self.probability = [value / sum_value in value in values]

    # def end(self, session):
    #         print(
    #         "ndcg@1:{}, hit@1:{}， ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ap:{}, valid_user:{}".
    #         format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
    #                self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
    #                self.ndcg_10 / self.valid_user,
    #                self.hit_10 / self.valid_user, self.ap / self.valid_user,
    #                self.valid_user))   
    
    # def before_run(self, run_context):
    #     variables = tf.get_collection('eval_sp')
    #     return tf.train.SessionRunArgs(variables)

    # def after_run(self, run_context, run_values):
    #     masked_lm_log_probs, input_ids, masked_lm_ids, info = run_values.results
    #     masked_lm_log_probs = masked_lm_log_probs.reshape((-1, max_predictions_per_seq, masked_lm_log_probs.shape[1]))
    

    #     for idx in range(len(input_ids)):
    #         rated = set(input_ids[idx])
    #         rated.add(0)
    #         rated.add(masked_lm_ids[idx][0])
    #         map(lambda x: rated.add(x), self.user_history['user_'+str(info[idx][0])][0])
    #         item_idx = [masked_lm_ids[idx][0]]

    #         masked_lm_log_probs_elem = masked_lm_log_probs[idx, 0]
    #         size_of_prob = len(self.ids) + 1

    #         if user_pop_random:
    #             while len(item_idx) < 101:
    #                 sampled_ids = np.random.choice(self.ids, 101, replace=False, p=self.probability)
    #                 sampled_ids = [x for x in sampled_ids if x not in rated and x not in item_idx]
    #                 item_idx.extend(sampled_ids[:])
    #             item_idx = item_idx[:101]
            
    #         else:
    #             for _ in range(100):
    #                 t = np.random.randint(1, size_of_prob)
    #                 while t in rated:
    #                     t = np.random.randint(1, size_of_prob)
    #                 item_idx.append(t)
            
    #         predictions = -masked_lm_log_probs_elem[item_idx]
    #         rank = predictions.argsort().argsort()[0]

    #         self.valid_user += 1

    #         if self.valid_user %100 == 0:
    #             print('.', end='')
    #             sys.stdout.flush()
            
    #         if rank < 1:
    #             self.ndcg_1 += 1
    #             self.hit_1 += 1
            
    #         if rank < 5:
    #             self.ndcg_5 += 1 / np.log2(rank+2)
    #             self.hit_5 += 1

    #         if rank < 10:
    #             self.ndcg_10 += 1 / np.log2(rank+2)
    #             self.hit_10 += 1
            
    #         self.ap += 1.0/(rank+1)
######################## here ############################


#%%

def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """
    返回 input_fn对象
    """
    def _decode_record(record, name_to_features):
        """
        int64 --> int32
        数据结构: d由: 
                    n 个 Tensor("IteratorGetNext_xxxx:0", shape=(), dtype=string) 组成
                    通过 tf.parse_to_example解析
                    每个 Tensor 解析成 1 个字典, 包含:
                        {'info': <tf.Tensor 'ParseSingleExample_xxx/ParseSingleExample:0' shape=(1,) dtype=int64>, 
                        'input_ids': <tf.Tensor 'ParseSingleExample_xxx/ParseSingleExample:1' shape=(50,) dtype=int64>, 
                        'input_lm_ids': <tf.Tensor 'ParseSingleExample_xxx/ParseSingleExample:2' shape=(30,) dtype=int64>, 
                        'input_lm_positions': <tf.Tensor 'ParseSingleExample_xxx/ParseSingleExample:3' shape=(30,) dtype=int64>, 
                        'input_lm_weight': <tf.Tensor 'ParseSingleExample_xxx/ParseSingleExample:4' shape=(30,) dtype=float32>, 
                        'input_mask': <tf.Tensor 'ParseSingleExample_xxx/ParseSingleExample:5' shape=(50,) dtype=int64>}
        """
        example = tf.parse_single_example(record, name_to_features)

        ## 将 int64 -> int32
        for name in example:
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        
        return example

    def input_fn(params):
        """
        从input_files导入数据, 并创建
        """
        batch_size = params['batch_size']
        
        name_to_features = {
            'info':tf.FixedLenFeature([1], tf.int64), #[user]
            'input_ids':tf.FixedLenFeature([max_seq_length], tf.int64),
            'input_mask':tf.FixedLenFeature([max_seq_length], tf.int64),
            'input_lm_positions':tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            'input_lm_ids':tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            'input_lm_weight':tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
            }

        ## 数据, 复制, 打乱, 处理, batch
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
        d = d.map(lambda record:_decode_record(record, name_to_features),
                  num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        
        return d

    return input_fn


#%%

def model_fn_builder(bert_config,
                     init_checkpoint,
                     learning_rate,
                     num_train_steps,
                     num_warmup_steps,
                     item_size):
    """
    返回一个model_fn 对象
    """

    def metric_fn(masked_lm_example_loss,
                  masked_lm_log_probs,
                  masked_lm_ids,
                  masked_lm_weights):
        """
        计算 model 的 accuracy 和 loss
        """
        
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs[-1]])
        # 每个label的预测
        masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(tf.reshape(masked_lm_example_loss, [-1]))
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])

        # accuracy
        masked_lm_accuracy = tf.metric.accuracy(labels=masked_lm_ids,
                                                predictions=masked_lm_predictions,
                                                weights=masked_lm_weights)
        # mean loss
        masked_lm_mean_loss = tf.metric.mean(values=masked_lm_example_loss, 
                                             weights=masked_masked_lm_weights)

        return {'masked_lm_accuracy':masked_lm_accuracy,
                'masked_lm_loss':masked_lm_mean_loss}



    
    #### Emebedding_layer ####
    def gather_indexes(sequence_tensor, positions):
        """
        保留带有位置信息的元素
        返回2d tensor , shape=[b*s, e]
        """
        # dim list, 例如[1, 5, 128] batch为1, seq_len为5, embedding为128
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        # [b * s, 1]
        #tf.range 生成一个size为batch_size的一维tensor, shape=(batch_size,)
        # 然后reshape为二维tensor, shape=(10, 1)(必须保证第二维度为1, -1维度就是剩下的维度值)
        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])

        # 叠加position, 还愿shape=[b*s, 1]
        flat_positions = tf.reshape(positions + flat_offsets, [-1])

        # [b, s, e] -> [b*s, e]
        flat_seq_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])

        # [b*s, e] -切片-[b*s,] --> [切片后的b*s, e]
        # 例如[[1], [2], [3], [4], [5]], [1, 2, 3] --> [[1], [2], [3]]
        output_tensor = tf.gather(flat_seq_tensor, flat_positions)
        return output_tensor

    
    def get_masked_lm_output(bert_config,
                             input_tensor,
                             output_weights,
                             positions,
                             label_ids,
                             label_weights):

        """
        获取Masked位置商品的 loss 和 log prob 
        """
        # return::input_tesnor[对应位置的b*s, e]
        input_tensor = gather_indexes(input_tensor, positions)

        with tf.variable_scope('cls/predictions'):
            # [b*s, e] --> [b*s, h]  (h=64)
            # wx+b
            input_tensor = tf.layers.dense(input_tensor,
                                           units=bert_config.hidden_size,
                                           activation=modeling.get_activation(bert_config.hidden_act),  ## glue
                                           kernel_constraint=modeling.create_initializer(bert_config.initializer_range))

            input_tesno = modeling.layer_norm(input_tensor)

        output_bias = tf.get_variable('output_bias',
                                      shape=[output_weights.shape[0]],
                                      initializer=tf.zeros_initializer())
        
        ## P(v) = softmax(gule(wx+b)E + b)
        # E 是 embedding_table
        # (原文) We use the shared item embedding matrix in the input and output layer
        # for alleviating overfitting and reducing model
        # [b*s, h] * [vocab_size, h]^T --> [b*s, vocab_size]
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias) # 加bias
        log_probs = tf.nn.log_softmax(logits, -1) # 归一化

        # tensor 化
        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        # one_hot_maxtrix, shape=[label_ids, h] 
        one_hot_labels = tf.one_hot(label_ids, 
                                   depth=output_weights.shape[0], 
                                   dtype=tf.float32)

        # position tensor 有可能有0 作为padding
        # label_weights : 0 表示padding, 1: 表示real predictions
        per_example_loss = - tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        # loss = ∑(w * ∑log_prob) / ∑w   其中w为[0,1]向量
        loss = numerator / denominator

        return (loss, per_example_loss, log_probs)



    def model_fn(features, labels, mode, params):

        print('*** features ***') 
        for name in features:
           print('name={}, shape={}'.format(name, features[name].shape))
        
        
        info = features['info']
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        masked_lm_positions = features['input_lm_positions']
        masked_lm_ids = features['input_lm_ids']
        masked_lm_weights = features['input_lm_weight']

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(config=bert_config,
                                  is_training=is_training, # true 的话 将 attention_drop 和 hidden_drop(送入dnn前drop, 送出dnn后drop, 和 合并input_embedding时) -> 0, 
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  token_type_ids=None, # 不进行上下句预测任务
                                  use_one_hot_embeddings=False) # 不用TPU, one_hot不建议点
        
        (masked_lm_loss, 
        masked_lm_example_loss,
        masked_lm_log_probs) = get_masked_lm_output(bert_config=bert_config,
                                                    input_tensor=model.get_sequence_output(),
                                                    output_weights=model.get_embedding_table(),
                                                    positions=masked_lm_positions,
                                                    label_ids=masked_lm_ids,
                                                    label_weights=masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_variable_names = {}

        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        

        print('**** Trainable Variables ****')

        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
                print('name=%s, shape=%s%s', var.name, var.shape, init_string)
        
        ## 训练与预测
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss=total_loss,
                                                     init_lr=learning_rate,
                                                     num_train_steps=num_train_steps,
                                                     num_warmup_steps=num_warmup_steps,
                                                     use_tpu=False)

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     train_op=train_op)
            

        elif mode == tf.estimator.ModeKeys.EVAL:

            # 将数据存入collection中
            tf.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.add_to_collection('eval_sp', input_ids)
            tf.add_to_collection('eval_sp', masked_lm_ids)
            tf.add_to_collection('eval_sp', info)

            ## 评估策略, 返回accuracy和均值loss的字典
            eval_matric = metric_fn(mask_lm_example_loss=mask_lm_example_loss,
                                    masked_lm_log_probs=masked_lm_log_probs,
                                    masked_lm_ids=masked_lm_ids,
                                    masked_lm_weights=masked_lm_weights)           

            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     loss=total_loss,
                                                     eval_metric_ops=eval_matric)
        
        return output_spec
    return model_fn

#%%

"""
测试用函数
"""
################################################################################################
def test(input_files):

    d = tf.data.TFRecordDataset(input_files)

    name_to_features = {
            'info':tf.FixedLenFeature([1], tf.int64), #[user]
            'input_ids':tf.FixedLenFeature([max_seq_length], tf.int64),
            'input_mask':tf.FixedLenFeature([max_seq_length], tf.int64),
            'input_lm_positions':tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            'input_lm_ids':tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            'input_lm_weight':tf.FixedLenFeature([max_predictions_per_seq], tf.float32)
            }

    for i in d:
        example = tf.parse_single_example(i, name_to_features)
        print(example)        
###############################################################################################
#%%
def main(*args):

    print(checkpointDir)
    if not do_train and not do_eval:
        print('you must do somthing')
        exit(1)

    ## TFRecord 文件加载
    train_input_files = []
    test_input_files = []
    train_input_files.append(train_input_file)
    test_input_files.append(test_input_file)
    print('train_input_file: %s' % (train_input_file))
    print('test_input_file: %s' % (test_input_file))

    ## 创建词表用于获取词的index, 从而得到word embedding
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f, encoding='utf-8')
    item_size = len(vocab.counter)


    ## 测试用
    # test(train_input_files)


    ## 创建 input_fn(Train or Eval)
    train_input_fn = input_fn_builder(input_files=train_input_files,
                                       max_seq_length=max_seq_length,
                                       max_predictions_per_seq=max_predictions_per_seq,
                                       is_training=True)

    eval_input_fn = input_fn_builder(input_files=test_input_files,
                                       max_seq_length=max_seq_length,
                                       max_predictions_per_seq=max_predictions_per_seq,
                                       is_training=False)

    # ## 创建 estimator Runconfig文件(设置阶段ckpt保存数)
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=save_checkpoints_steps)
    
    
    # ## 创建model_fn 和 bert参数初始化
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    model_fn = model_fn_builder(bert_config=bert_config,
                                init_checkpoint=init_checkpoint,
                                learning_rate=learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                item_size=item_size)

    eval_input_fn = model_fn_builder(bert_config=bert_config,
                                     init_checkpoint=init_checkpoint,
                                     learning_rate=learning_rate,
                                     num_train_steps=num_train_steps,
                                     num_warmup_steps=num_warmup_steps,
                                     item_size=item_size)    

    # ## 创建 estimator
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={'batch_size':batch_size})



    ## 进行 trian or evaluate
    # train
    if do_train:
        print('##### Running training #####')
        print('Batch_size=%s' % (batch_size))
        estimator.train(input_fn=train_input_fn,
                        max_steps=10000)


    # eval
    if do_eval:
        print('##### Running evaluation #####')
        print('Batch_size=%s' % (batch_size))
        result = estimator.evaluate(input_fn=eval_input_fn,
                           step=None)

    # 保存预测结果
    ouput_eval_file = os.path.join(checkpointDir, 'eval_result.txt')
    with tf.gfile.GFile(ouput_eval_file, 'w') as writer:
        print('***** Eval results *****')
        print(bert_config.to_json_string())
        writer.write(bert_config.to_json_string()+'\n')
        for key in sorted(result.key()):
            print('%s = %s' , key, str(result[key]))
            writer.write('%s = %s\n' % (key, str(result[key])))


    print('DONE!!!')






#%%

if __name__ == "__main__":

    signature = 'beauty_exp_2020-7-7'

    bert_config_file = './config_file/bert_config_beauty_64.json'
    train_input_file = './data/beauty_hao-2020-7-7.train.tfrecord'
    test_input_file = './data/beauty_hao-2020-7-7.test.tfrecord'
    checkpointDir = './checkpoints/'+signature
    signature = signature
    init_checkpoint = None
    tpu_name = None
    tup_zone = None
    gcp_project = None
    num_tpu_core = None
    
    vocab_filename = './data/beauty_hao-2020-7-7.vocab'
    user_history_filename = './data/beauty_hao-2020-7-7.his'
    
    max_seq_length = 50
    max_predictions_per_seq = 30
    batch_size = 256
    learning_rate = 1e-4

    num_train_steps = 400000
    num_warmup_steps = 100
    save_checkpoints_steps = 10000
    iterations_per_loop = 1000
    max_eval_steps = 1000


    use_tpu = False
    do_train = True
    do_eval = True
    user_pop_random = False
    
    
    main(checkpointDir, do_train, do_eval, bert_config_file,
          train_input_file, test_input_file, save_checkpoints_steps,
          batch_size)    