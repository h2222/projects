import tensorflow as tf
import os
import sys
sys.path.append('test')
import numpy as np
import pandas as pd
import copy
from nltk.corpus import stopwords
from nltk import tokenize
from gensim.models import KeyedVectors
sw = stopwords.words('portuguese')


def load_textcnn_with_embedding_table_model_file(pb_path):
    with tf.Graph().as_default() as graph:
        tf.train.import_meta_graph(pb_path)
    return graph


def model_predict(graph, X_test, Y_test):
    with tf.Session(graph=graph) as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        input_vec = sess.graph.get_tensor_by_name("Input_sent:0")
        prediction = sess.graph.get_tensor_by_name("Output:0")
        feed_dict = {input_vec:X_test}
        prob = sess.run(prediction, feed_dict)
    return prob, Y_test


def data_process(text_batch, sentence_len, w2v_vocab):
    res_content = []
    for content in text_batch:
        try:
            word_ls =  tokenize.word_tokenize(content, language='portuguese') 
            word_ls = [i for i in word_ls if (i not in sw) and (i in w2v_vocab)]
            while (len(word_ls) != sentence_len):
                if len(word_ls) > sentence_len:
                    word_ls = word_ls[:sentence_len]
                elif len(word_ls) < sentence_len:
                    word_ls.append("<PAD>")
            res_content.append(word_ls)
        except Exception as e:
            print(e)
            continue
    res_content = np.array(res_content)
    return res_content


def test_model(bp_path, w2v_path):
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path)
    graph = load_textcnn_with_embedding_table_model_file(bp_path)
    vec_dim = w2v_model.vector_size
    total_vocab_size = len(w2v_model.vocab)
    sorted_vocab = sorted(w2v_model.vocab)

    test_no_trip_60_case_f = open('test/total_case_v4_no_change.csv', 'r')
    test_no_trip_60_case_f.readline()

    # test_no_trip_200_case
    test_no_trip_200_case_f = open('test/total_case_v4_no_change.csv', 'r')
    test_no_trip_200_case_f.readline()

    bad_case_path = 'test/bad_case_v4.csv'
    good_case_path = 'test/good_case_v4.csv'
    open(bad_case_path, 'w')
    open(good_case_path, 'w')

    for i, line in enumerate(test_no_trip_200_case_f.readlines()):
        if i == -1:
            break
        line_ls = line.split('|')
        ticket_id = line_ls[0]
        content = line_ls[3]
        semantic_label = line_ls[-1]
        trans = line_ls[4]

        x = np.array([line_ls[3]])
        y = np.array([[1, 0]])
        x = data_process(x, 30, sorted_vocab)
        prob, Y_test = model_predict(graph, x, y)
        print('----')

        if prob[0, 0] > prob[0, 1]:
            print('good case')
            with open(good_case_path, 'a') as f:
                good_case = ticket_id + "|" + content + "|" + trans + "|" + str(prob[-1, 0]) + "|" + semantic_label
                print(good_case)
                f.write(good_case)
        else:
            print('bad case')
            with open(bad_case_path, 'a') as f:
                bad_case = ticket_id + "|" + content + "|" + trans + "|" + str(prob[-1, -1]) + "|" + semantic_label 
                print(bad_case)
                f.write(bad_case)


if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    test_model(path1, path2)
    # print_graph(path2)