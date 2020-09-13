import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
import os


path_model_1_dir = '/home/zengbin.gao/multi_Alstm_classify-gzb/out_put/out_put_03_12_01'
path_model_1_meta = path_model_1_dir + '/' + 'MyModel.meta'
path_model_1_index = path_model_1_dir + '/' + 'MyModel.index'
path_model_1_data = path_model_1_dir + '/' + 'MyModel.data-00000-of-00001'

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

sess = tf.Session()

model_1 = tf.train.import_meta_graph(path_model_1_meta)
model_1.restore(sess, tf.train.latest_checkpoint(path_model_1_dir))
graph = tf.get_default_graph()

builder = saved_model_builder.SavedModelBuilder(path_model_1_dir + '/pb')

x = graph.get_tensor_by_name("Input/input_x:0")
seqlen = graph.get_tensor_by_name("Input/input_seqlen:0")
keep_prob = graph.get_tensor_by_name("Input/keep_prob:0")

prediction_ask_know = graph.get_tensor_by_name("ask_know/Prediction/prediction:0")
prediction_identity = graph.get_tensor_by_name("identity/Prediction/prediction:0")
#prediction_dealh = graph.get_tensor_by_name("dealh/Prediction/prediction:0")
#prediction_dealml = graph.get_tensor_by_name("dealml/Prediction/prediction:0")
prediction_ask_today = graph.get_tensor_by_name("ask_today/Prediction/prediction:0")
prediction_ask_tomorrow = graph.get_tensor_by_name("ask_tomorrow/Prediction/prediction:0")
prediction_once = graph.get_tensor_by_name("once/Prediction/prediction:0")
prediction_request = graph.get_tensor_by_name("request/Prediction/prediction:0")
prediction_twice = graph.get_tensor_by_name("twice/Prediction/prediction:0")

inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x),
          'input_seqlen': tf.saved_model.utils.build_tensor_info(seqlen),
          'input_keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)}
          
outputs = {'output_ask_know': tf.saved_model.utils.build_tensor_info(prediction_ask_know),
           'output_identity': tf.saved_model.utils.build_tensor_info(prediction_identity),
           'output_ask_today': tf.saved_model.utils.build_tensor_info(prediction_ask_today),
           'output_ask_tomorrow': tf.saved_model.utils.build_tensor_info(prediction_ask_tomorrow),
           'output_once': tf.saved_model.utils.build_tensor_info(prediction_once),
           'output_request': tf.saved_model.utils.build_tensor_info(prediction_request),
           'output_twice': tf.saved_model.utils.build_tensor_info(prediction_twice)}
#outputs = {'output_ask_know': tf.saved_model.utils.build_tensor_info(prediction_ask_know),
#           'output_identity': tf.saved_model.utils.build_tensor_info(prediction_identity),
#           'output_dealh': tf.saved_model.utils.build_tensor_info(prediction_dealh),
#           'output_dealml': tf.saved_model.utils.build_tensor_info(prediction_dealml)}
#outputs = {'output_ask_know': tf.saved_model.utils.build_tensor_info(prediction_ask_know),
#           'output_identity': tf.saved_model.utils.build_tensor_info(prediction_identity),
#           'output_dealh': tf.saved_model.utils.build_tensor_info(prediction_dealh),
#           'output_dealml': tf.saved_model.utils.build_tensor_info(prediction_dealml),
#           'output_ask_today': tf.saved_model.utils.build_tensor_info(prediction_ask_today),
#           'output_ask_tomorrow': tf.saved_model.utils.build_tensor_info(prediction_ask_tomorrow),
#           'output_once': tf.saved_model.utils.build_tensor_info(prediction_once),
#           'output_request': tf.saved_model.utils.build_tensor_info(prediction_request),
#           'output_twice': tf.saved_model.utils.build_tensor_info(prediction_twice)}

signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

builder.add_meta_graph_and_variables(
    sess=sess,
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
builder.save()
                                                                                                         