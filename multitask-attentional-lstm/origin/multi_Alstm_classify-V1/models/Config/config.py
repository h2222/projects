"""
Encoding: utf-8
Author: April
Email: imlpliang@gmail.com
CreateTime: 2019-08-21 16:18
Description: config.py
Version: 1.0
"""
from configparser import ConfigParser
import os


class Configurable(object):
    def __init__(self, config_file, extra_args):
        # 实例化ConfigParser对象
        config = ConfigParser()
        # 读取文件
        config.read(config_file, encoding='utf-8')
        if extra_args:
            extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
        # 如果模型参数已存在
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
        # 创建目录
        self._config = config
        print("self.save_dir",config_file)
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
        # 记录参数 写入文件
        config.write(open(self.config_path, 'w'))
        print('Loaded config file sucessfully!')
        for section in config.sections():
            for k, v in config.items(section):
                print("{} = {}".format(k, v))

    def add_args(self, key, value):
        self._config.set(self.add_sec, key, value)
        self._config.write(open(self.config_file, 'w'))

    def read_args(self):
        dict_args = {'Sections': [], 'Key': [], 'Values': []}
        for section in self._config.sections():
            for k, v in self._config.items(section):
                dict_args['Sections'].append(section)
                dict_args['Key'].append(k)
                dict_args['Values'].append(v)
        return dict_args

    # [Data]
    @property
    def data_dir(self):
        return self._config.get('Data', 'data_dir')


    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def max_len(self):
        return self._config.getint('Data', 'max_len')

    @property
    def min_len_word(self):
        return self._config.getint('Data', 'min_len_word')

    @property
    def concat_in_node(self):
        return self._config.getboolean('Data', 'concat_in_node')


    @property
    def segment(self):
        return self._config.get('Data', 'segment')

    @property
    def cws_model_path(self):
        return self._config.get('Data', 'cws_model_path')

    @property
    def classification_combine(self):
        return self._config.getboolean('Data', 'classification_combine')

    @property
    def use_cols(self):
        this_usecols = self._config.get('Data', 'use_cols')
        this_usecols = [x for x in this_usecols.split(',')]
        return this_usecols

    @property
    def label_maps_name(self):
        label_maps_name = eval(self._config.get('Data', 'label_maps'))
        label_maps_name = [x for x in label_maps_name.split(',')]
        return label_maps_name

    @property
    def filter_invalid(self):
        return self._config.getboolean('Data', 'filter_invalid')

    # [Save]
    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def tokenizer_path(self):
        return self._config.get('Save', 'tokenizer_path')

    @property
    def config_path(self):
        return self._config.get('Save', 'config_path')

    @property
    def model_path(self):
        return self._config.get('Save', 'model_path')

    @property
    def save_model_name(self):
        return self._config.get('Save', 'save_model_name')

    @property
    def checkpoint_num(self):
        return self._config.getint('Save', 'checkpoint_num')

    @property
    def checkpoint_from_epoch(self):
        return self._config.getint('Save', 'checkpoint_from_epoch')

    @property
    def train_sample(self):
        return self._config.get('Save', 'train_sample')


    @property
    def test_sample(self):
        return self._config.get('Save', 'test_sample')

    # [Model]
    @property
    def vocab_size(self):
        return self._config.getint('Model', 'vocab_size')

    @vocab_size.setter
    def vocab_size(self, value):
        self._config.set('Model', 'vocab_size', str(value))

    @property
    def n_class(self):
        return self._config.getint('Model', 'n_class')

    @n_class.setter
    def n_class(self, value):
        self._config.set('Model', 'n_class', str(value))

    @property
    def embedding_size(self):
        return self._config.getint('Model', 'embedding_size')

    @property
    def hidden_size(self):
        return self._config.getint('Model', 'hidden_size')

    @property
    def filter_sizes(self):
        value = self._config.get("Model", "filter_sizes")
        value = [int(k) for k in list(value) if k != ","]
        return value

    @property
    def num_filters(self):
        return self._config.getint('Model', 'num_filters')

    @property
    def dropout_keep_prob(self):
        return self._config.getfloat('Model', 'dropout_keep_prob')

    @property
    def learning_rate(self):
        return self._config.getfloat('Model', 'learning_rate')

    @property
    def l2_reg_lambda(self):
        return self._config.getfloat('Model', 'l2_reg_lambda')
    
    @property
    def keep_prob(self):
        return self._config.getfloat('Model', 'keep_prob')
    
    @property
    def num_layers(self):
        return self._config.getint('Model', 'num_layers')

    

    # [Run]
    @property
    def is_train(self):
        return self._config.getboolean('Run', 'is_train')

    @property
    def train_epochs(self):
        return self._config.getint('Run', 'train_epochs')

    @property
    def train_steps(self):
        return self._config.getint('Run', 'train_steps')

    @property
    def batch_size(self):
        return self._config.getint('Run', 'batch_size')

    @property
    def evaluate_every(self):
        return self._config.getint('Run', 'evaluate_every')
    
    @property
    def cls_type(self):
        this_cls_type = self._config.get('Data', 'cls_type')
        this_cls_type = [x for x in this_cls_type.split(',')]
        return this_cls_type
    
    @property
    def label_map_msg(self):
        this_label_map_msg = self._config.get('Data', 'label_map_msg')
        this_label_map_msg = [x for x in this_label_map_msg.split('||')]
        return this_label_map_msg

    @property
    def path_model_Predict(self):
        return self._config.get('Predict', 'path_model_Predict')

    @property
    def name_model_Predict(self):
        return self._config.get('Predict', 'name_model_Predict')

