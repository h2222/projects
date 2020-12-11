#coding=utf-8
"""
现在不用这种方式了, 太痛苦了!!!

"""
import argparse
from configparser import ConfigParser



class Configurable:
    def __init__(self, config_file, extra_args):
        config = ConfigParser()
        config.read(config_file, encoding='utf-8')

        # 修改参数值(根据extra参数数值)
        for section in config.sections():
            print(section)
            for key, value in config.items(section):
                if key in extra_args:
                    value = type(value)(extra_args[key])
                    config.set(section, key, value)

        # 创建配置器对象
        self._config = config
        
        # 创建目录
        print("配置文件保存路径:%s" % self.save_dir)
        if not os.path.isdir(self.save_dir): os.mkdir(self.save_dir)

        # 记录参数 写入文件
        config.write(open(self.config_path, 'w'))

    def add_args(self, section, key, value):
        self._config.set(section, key, value)
        self._config.write(open(self.save_dir, 'w', encoding='utf-8'))
    
    def read_args(self):
        dict_args = {'Sections': [], 'Key': [], 'Values': []}
        for section in self._config.sections():
            for k, v in self._config.items(section):
                dict_args['Sections'].append(section)
                dict_args['Key'].append(k)
                dict_args['Values'].append(v)
        return dict_arg


    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def tmp_file_path_dir(self):
        return self._config.get('Save', 'tmp_file_path_dir')



if __name__ == "__main__":
    # python3 config.py --test_add x --test_add2 x2
    argparser = argparse.ArgumentParser(description="Neural network parameters")
    argparser.add_argument('--config_file', default='/Users/didi/Documents/projects/multitask-attentional-lstm/origin/multi_Alstm_classify-V1/models/Config/config.cfg')
    # argparser.add_argument('--test_add', default='add') 
    args, extra_args = argparser.parse_known_args() # extra_args 解析未识别的参数
    

    if extra_args:
        extra_args = dict([(k[2:], v) for k, v in zip(extra_args[0::2], extra_args[1::2])])
    print(args.config_file)
    print(extra_args)
    configable(args.config_file, extra_args)

    # print(extra_args)
