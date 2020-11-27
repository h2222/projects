# -*- coding: utf-8 -*-


class ProcessIdNotFoundError(AttributeError):
    def __init__(self, pid):
        super(ProcessIdNotFoundError, self).__init__("语义理解流程: {pid} 未在模型中定义...".format(pid=pid))