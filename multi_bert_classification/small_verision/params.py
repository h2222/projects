import sys, os



class Param:

    def __init__(self):
        self.run_task_name_list = []

        # train file path
        # 存放任务数据临时文件
        self.tmp_dir = 'tmp'
        if not os.path.isdir(self.tmp_dir):
            os.makedir(self.tmp_dir)



        self.task_type = {'task1':'cls', 'task2':'cls'}

        # 特殊字符
        self.TRAIN = 'train'
        self.EVAL = 'eval'
        self.PREDICT = 'predict'
        
        # 任务字符
        # 任务类型 pretrain预训练(False) 与 cls 分类(True)
        self.pre_train = 'pretrain'
        self.cls = 'cls'
