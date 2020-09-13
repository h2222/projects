import sys
import time
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from utils.model_helper import *
import os
import argparse
import Config.config as configurable
from utils.data_helper import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


random.seed(seed_num)


class ABLSTM(object):
    def __init__(self, config):
        self.max_len = config.max_len
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.cls_type = config.cls_type
        
        print('class', self.n_class)
        print('cls_type', self.cls_type)
        # placeholder
        with tf.name_scope("Input"):
            self.x = tf.placeholder(tf.int32, [None, self.max_len], name='input_x')
            self.seqlen = tf.placeholder(tf.int32, [None], name='input_seqlen')
            self.label = tf.placeholder(tf.int32, [None], name='input_y')
            self.keep_pRob = tf.placeholder(tf.float32, name='keep_prob')

    def lstm_cell_with_dropout(self,state_size,keep_prob):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units = state_size)
        cell = tf.contrib.rnn.DropoutWrapper(
                                            cell = cell,
                                            output_keep_prob = keep_prob,
                                            state_keep_prob = keep_prob,
                                            variational_recurrent = True,
                                            dtype = tf.float32)
        return cell
    def blstm_layer_with_dropout(self, inputs, seqlen, state_size, keep_prob, scope):
        cell1 = self.lstm_cell_with_dropout(state_size, keep_prob)
        cell2 = self.lstm_cell_with_dropout(state_size, keep_prob)
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                                                                   cell_fw = cell1,
                                                                   cell_bw = cell2,
                                                                   inputs = inputs,
                                                                   sequence_length = seqlen,
                                                                   dtype = tf.float32,
                                                                   scope = scope)
        return tf.concat([output_fw, output_bw],axis = -1)       
    def build_graph(self):
        print("building graph")
        # Word embedding
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        rnn_inputs = tf.nn.embedding_lookup(embeddings_var, self.x)

        for i in range(config.num_layers):
            with tf.name_scope("BLSTM-{}".format(i)) as scope:
                rnn_inputs = self.blstm_layer_with_dropout(rnn_inputs, 
                    self.seqlen, self.hidden_size, self.keep_pRob, scope)
        H = rnn_inputs
        M = tf.tanh(H)
        W = tf.Variable(tf.random_normal([self.hidden_size*2], stddev=0.1))
        self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, self.hidden_size*2]),
                                                        tf.reshape(W, [-1, 1])),
                                              (-1, self.max_len)))  # batch_size x seq_len
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(self.alpha, [-1, self.max_len, 1]))
        r = tf.squeeze(r)
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        h_drop = tf.nn.dropout(h_star, self.keep_pRob)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        for item in self.cls_type:
            with tf.variable_scope(item):
                FC_W = tf.Variable(tf.truncated_normal([self.hidden_size*2, self.n_class], stddev=0.1))
                FC_b = tf.Variable(tf.constant(0., shape=[self.n_class]))
                y_hat = tf.nn.xw_plus_b(h_drop, FC_W, FC_b)

                # loss
                setattr(self, 'loss_'+item, tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label)))
               
                # prediction
                with tf.name_scope("Prediction"):
                    setattr(self, 'prediction_'+item, tf.argmax(tf.nn.softmax(y_hat), 1, name='prediction'))

                loss_to_minimize = getattr(self,'loss_'+item)
                tvars = tf.trainable_variables()
                gradients = tf.gradients(loss_to_minimize, tvars,
                                         aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

                setattr(self,'train_'+item+'_op', self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                                         name='train_step'))

        print("graph built successfully!")

def classify_evaluation(test_predictions, target, id2type, class_name, cls_type):
    print(len(test_predictions))
    print(len(target))
    print(len(class_name))
    result = np.array(list(zip(test_predictions, target, class_name)))
    classifys = cls_type
    for classify in classifys:
        index_k = result[:, 2] == classify
        print(classify)
        my_evaluation(list(result[index_k][:, 0].astype(int)), list(result[index_k][:, 1].astype(int)), 0, id2type)
        
if __name__ == '__main__':
    start_time = time.time()
    print("Process ID: {}, Process Parent ID: {}".format(os.getpid(), os.getppid()))
    argparser = argparse.ArgumentParser(description="Neural network parameters")
    argparser.add_argument('--config_file', default='./models/Config/config.cfg')
   
    args, extra_args = argparser.parse_known_args()
    print("Config_File = {}\nExtra_Args = {}".format(args.config_file, extra_args))
    config = configurable.Configurable(args.config_file, extra_args)
    print("seed_num = {}".format(seed_num))

    print("\nLoading Data......")
    seq_info, dict_id2type = load_all_data(config)
    train_texts, train_y, class_train, seq_l_train, test_texts, test_y, class_test, seq_l_test = seq_info

    del seq_info
    train_x, test_x, vocab_size = data_preprocessing_v2(train_texts, test_texts, config.tokenizer_path, max_len=32)
    config.n_class = len(dict_id2type)
    config.vocab_size = vocab_size
    print("n_class = {}".format(config.n_class))
    print("vocab_size = {}".format(config.vocab_size))

    if not os.path.exists(config.save_dir):  # 如果路径不存在
        os.makedirs(config.save_dir)

    # 构建计算图&初始化
    classifier = ABLSTM(config)
    classifier.build_graph()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Train
    the_best_f = 0
    saver = tf.train.Saver(max_to_keep=config.checkpoint_num)
    start = time.time()
    for epoch in range(config.train_epochs):
        time4this_epoch_begin = time.time()
        print("\n#### Current Train Epoch: {} ####".format(epoch+1))
        for x_batch, y_batch, class_name, seq_l_batch, _ in fill_feed_dict(train_x, train_y, class_train, seq_l_train, train_texts, config.batch_size):

            step, loss, accuracy = run_train_step(classifier, sess, (x_batch, y_batch, seq_l_batch), class_name, config.dropout_keep_prob)
            attn = get_attn_weight(classifier, sess, (x_batch, y_batch, seq_l_batch), config.dropout_keep_prob)
            time_str = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
            if step % config.train_steps == 0:
                print("{}: step {}, loss {:.6f}, accuracy {:.6f}".format(time_str, step, loss, accuracy))
        print("Train Epoch Time: {:.6f} s".format(time.time() - time4this_epoch_begin))

        # Evaluation
        print("### Evaluation...")
        print("### Train")
        train_loss, train_count = 0, 0
        train_predictions, train_targets, train_texts_reals, train_classify_name = [], [], [], []
        for x_batch, y_batch, class_name, seq_l_batch, text_real_batch in fill_feed_dict(train_x, train_y, class_train, seq_l_train, train_texts, config.batch_size):
            result = run_eval_step(classifier, sess, class_name, (x_batch, y_batch, seq_l_batch))
            train_count += 1
            train_classify_name += [str(class_name)]*len(list(y_batch))
            train_loss += result[0]
            train_predictions += list(result[1])
            train_targets += list(y_batch)
            train_texts_reals += list(text_real_batch)
        train_accuracy = my_evaluation(train_predictions, train_targets, train_loss / train_count, dict_id2type)
        print("### Test")
        test_loss, test_count = 0, 0
        test_predictions, test_targets, test_texts_reals, test_classify_name= [], [], [], []
        for x_batch, y_batch, class_name, seq_l_batch, text_real_batch in fill_feed_dict(test_x, test_y, class_test, seq_l_test, test_texts, config.batch_size):
            result = run_eval_step(classifier, sess, class_name, (x_batch, y_batch, seq_l_batch))
            test_count += 1
            test_classify_name += [str(class_name)]*len(list(y_batch))
            test_loss += result[0]
            test_predictions += list(result[1])
            test_targets += list(y_batch)
            test_texts_reals += list(text_real_batch)
        print('all average result')
        test_f = my_evaluation(test_predictions, test_targets, test_loss / test_count, dict_id2type)
        classify_evaluation(test_predictions, test_targets, dict_id2type, test_classify_name, config.cls_type)

        # Save
        if test_f > the_best_f and epoch+1 > config.checkpoint_from_epoch:
            the_best_f = test_f
            print("The current best model.")
            save_forecast_sample(train_predictions, train_targets, train_texts_reals, dict_id2type, config.train_sample, train_classify_name, epoch)
            save_forecast_sample(test_predictions, test_targets, test_texts_reals, dict_id2type, config.test_sample, test_classify_name, epoch)
            print("Successfully saved all SampleInfo to {}".format(config.save_dir))
            # TODO:保存模型和样例
            saver.save(sess, config.save_dir + '/'+config.save_model_name)
            print("Successfully saved the model to {}".format(config.save_dir))


