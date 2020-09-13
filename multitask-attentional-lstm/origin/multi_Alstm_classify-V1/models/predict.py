import sys
import time
import os
import argparse
import Config.config as configurable
from utils.data_helper import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(seed_num)



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


def run_eval_step2(graph, sess, class_name, batch, dropout_keep_prob=1.0):
    x = graph.get_tensor_by_name("Input/input_x:0")
    y = graph.get_tensor_by_name("Input/input_y:0")
    seq_l = graph.get_tensor_by_name("Input/input_seqlen:0")
    keep_prob = graph.get_tensor_by_name("Input/keep_prob:0")
    feed_dict = {x: batch[0],
                 y: batch[1],
                 seq_l:batch[2],
                 keep_prob: dropout_keep_prob}

    prediction = graph.get_tensor_by_name(""+class_name+"/Prediction/prediction:0") 
    
    predictions = sess.run(prediction, feed_dict)
    return predictions



def data_preprocessing_predict(config,train, max_len):
    path_model_dir = config.save_dir
    path_model_tokenizer = path_model_dir + '/' + 'this_tokenizer.pickle'

    with open(path_model_tokenizer, 'rb') as handle:
        model_tokenizer = pickle.load(handle)

    train_model = model_tokenizer.texts_to_sequences(train)

    train_padded = pad_sequences(train_model, maxlen=max_len, padding='post', truncating='post')
    return train_padded

if __name__ == '__main__':
    start_time = time.time()
    print("Process ID: {}, Process Parent ID: {}".format(os.getpid(), os.getppid()))
    argparser = argparse.ArgumentParser(description="Neural network parameters")
    argparser.add_argument('--config_file', default='./Config/config.cfg')
    args, extra_args = argparser.parse_known_args()
    print("Config_File = {}\nExtra_Args = {}".format(args.config_file, extra_args))
    config = configurable.Configurable(args.config_file, extra_args)
    print("seed_num = {}".format(seed_num))

    print("\nLoading Data......")
    seq_info, dict_id2type = load_all_data(config)
    predict_texts, predict_y, class_predict, seq_l_predict = seq_info

    del seq_info
    predict_x= data_preprocessing_predict(config, predict_texts,  max_len=32)
    config.n_class = len(dict_id2type)
    print("n_class = {}".format(config.n_class))

    sess = tf.Session()

    path_model_1_dir = config.path_model_Predict
    path_model_1_meta = path_model_1_dir + '/' +config.name_model_Predict+'.meta'
    path_model_1_index = path_model_1_dir + '/' + config.name_model_Predict+'.index'
    path_model_1_data = path_model_1_dir + '/' + config.name_model_Predict+'.data-00000-of-00001'
    
    model_1 = tf.train.import_meta_graph(path_model_1_meta)
    model_1.restore(sess, tf.train.latest_checkpoint(path_model_1_dir))
    graph = tf.get_default_graph()

    predict_loss, predict_count = 0, 0
    predict_predictions, predict_targets, predict_texts_reals, predict_classify_name= [], [], [], []
    for x_batch, y_batch, class_name, seq_l_batch, text_real_batch in fill_feed_dict(predict_x, predict_y, class_predict, seq_l_predict, predict_texts, config.batch_size):
        result = run_eval_step2(graph, sess, class_name, (x_batch, y_batch, seq_l_batch))
        predict_count += 1
        predict_classify_name += [str(class_name)]*len(list(y_batch))
        predict_predictions += list(result)
        predict_targets += list(y_batch)
        predict_texts_reals += list(text_real_batch)
    print('all average result')
    test_f = my_evaluation(predict_predictions, predict_targets, predict_loss / predict_count, dict_id2type)
    classify_evaluation(predict_predictions, predict_targets, dict_id2type, predict_classify_name, config.cls_type)

    save_forecast_sample(predict_predictions, predict_targets, predict_texts_reals, dict_id2type, config.test_sample, predict_classify_name, 'predict')
    print("Predict end!!")


