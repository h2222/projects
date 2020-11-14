#coding=utf-8
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


ckpt_fname = 'model.ckpt-2000.meta'
graph = tf.get_default_graph()
loader = tf.train.import_meta_graph(ckpt_fname)
#loader.restore(sess, tf.train.latest_checkpoint('.'))
sess = tf.Session(graph=graph)

# get all ops and tensors params names
# name = [n.name for n in graph.as_graph_def().node if 'word_embeddings' in n.name]

# check all vriable names
lt =  tf.train.latest_checkpoint('.')
print(tf.train.list_variables(lt))

# print all tensor values
#print_tensors_in_checkpoint_file(lt, all_tensors=True, tensor_name='')

## load vriable
word_table = tf.train.load_variable(lt, 'bert/embeddings/word_embeddings')

print(word_table)
print(type(word_table), word_table.shape)

# for tensorboard
#ile_writer = tf.summary.FileWriter(logdir='checkpoint_log_dir/faceboxes', graph=graph)
