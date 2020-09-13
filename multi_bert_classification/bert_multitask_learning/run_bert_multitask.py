import time
import os
import argparse

import tensorflow as tf

from tensorflow.estimator import Estimator
from tensorflow.estimator import train_and_evaluate, TrainSpec, EvalSpec

from .input_fn import train_eval_input_fn, predict_input_fn
from .model_fn import BertMultiTask
from .params import BaseParams, DynamicBatchSizeParams
from .ckpt_restore_hook import RestoreCheckpointHook
from . import metrics
from .utils import TRAIN, EVAL, PREDICT
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _create_estimator(
        num_gpus=1,
        params=DynamicBatchSizeParams(),
        model=None):
    if model is None:
        ## 初始化模型
        ## 再次初始化模型(一次初始化早multi_classify.py中)
        model = BertMultiTask(params=params)
    
    # 获得model_fn，包括Bert模型的定义
    # 返回feature_dict keys 包括 seq，all，pooled， 每一次处理后的tensor
    model_fn = model.get_model_fn(warm_start=False)
    
    # tensorflow 分布式训练策略
    # https://blog.csdn.net/zwqjoy/article/details/89552866
    # 多线程同步策略，复制每一份变量下每一个使用的gpu上
    dist_trategy = tf.contrib.distribute.MirroredStrategy(
        num_gpus=int(num_gpus),
        cross_tower_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(
            'nccl', num_packs=int(num_gpus)))

    run_config = tf.estimator.RunConfig(
        train_distribute=dist_trategy,
        eval_distribute=dist_trategy,
        log_step_count_steps=params.log_every_n_steps)

    # ws = make_warm_start_setting(params)

    estimator = Estimator(
        model_fn,
        model_dir=params.ckpt_dir,
        params=params,
        config=run_config)
    return estimator



### 训练步骤！！！！
def train_bert_multitask(
        problem='weibo_ner',
        num_gpus=1,
        num_epochs=10,
        model_dir='',
        params=None,
        problem_type_dict={},
        processing_fn_dict={},
        model=None):
    """Train Multi-task Bert model

    About problem: 
        There are two types of chaining operations can be used to chain problems.
            - `&`. If two problems have the same inputs, they can be chained using `&`. Problems chained by `&` will be trained at the same time.
            - `|`. If two problems don't have the same inputs, they need to be chained using `|`. Problems chained by `|` will be sampled to train at every instance.

        For example, `cws|NER|weibo_ner&weibo_cws`, one problem will be sampled at each turn, say `weibo_ner&weibo_cws`, then `weibo_ner` and `weibo_cws` will trained for this turn together. Therefore, in a particular batch, some tasks might not be sampled, and their loss could be 0 in this batch.

    About problem_type_dict and processing_fn_dict:
        If the problem is not predefined, you need to tell the model what's the new problem's problem_type
        and preprocessing function.
            For example, a new problem: fake_classification
            problem_type_dict = {'fake_classification': 'cls'}
            processing_fn_dict = {'fake_classification': lambda: return ...}

        Available problem type:
            cls: Classification
            seq_tag: Sequence Labeling
            seq2seq_tag: Sequence to Sequence tag problem
            seq2seq_text: Sequence to Sequence text generation problem

        Preprocessing function example:
        Please refer to https://github.com/JayYip/bert-multitask-learning/blob/master/README.md

    Keyword Arguments:
        problem {str} -- Problems to train (default: {'weibo_ner'})
        num_gpus {int} -- Number of GPU to use (default: {1})
        num_epochs {int} -- Number of epochs to train (default: {10})
        model_dir {str} -- model dir (default: {''})
        params {BaseParams} -- Params to define training and models (default: {DynamicBatchSizeParams()})
        problem_type_dict {dict} -- Key: problem name, value: problem type (default: {{}})
        processing_fn_dict {dict} -- Key: problem name, value: problem data preprocessing fn (default: {{}})
    """
    if params is None:
        params = DynamicBatchSizeParams()

    if not os.path.exists('models'):
        os.mkdir('models')

    # 抓取模型路径和文件名
    if model_dir:
        base_dir, dir_name = os.path.split(model_dir)
    else:
        base_dir, dir_name = None, None
    params.train_epoch = num_epochs
    # add new problem to params if problem_type_dict and processing_fn_dict provided
    # 添加 问题名称， 问题类型，问题数据导入函数（如果有top和share）设置到params中
    if processing_fn_dict:
        ## 'ask_today': load_data of 'ask_today'
        for new_problem, new_problem_processing_fn in processing_fn_dict.items():
            print('Adding new problem {0}, problem type: {1}'.format(
                new_problem, problem_type_dict[new_problem]))

            params.add_problem(
                problem_name=new_problem, problem_type=problem_type_dict[new_problem], processing_fn=new_problem_processing_fn)
    
    
    # 分割问题list使用‘|’ 或‘&’， 创建ckpt保存位置， 计算总训练数据和划分训练步数
    params.assign_problem(problem, gpu=int(num_gpus),
                          base_dir=base_dir, dir_name=dir_name)
    
    # 参数json话，准备送入estimator
    params.to_json()


    # 创建 estimator ， params， model在multi_classify.py 设置为BertMultiTask(params)
    estimator = _create_estimator(
        num_gpus=num_gpus, params=params, model=model)

    train_hook = RestoreCheckpointHook(params)
    
    def train_input_fn(): return train_eval_input_fn(params)
    def eval_input_fn(): return train_eval_input_fn(params, mode=EVAL)
    
    
    # train_spec 是 tf.estimator.TrainSpec 对象，用于指定训练的输入函数以及其它参数
    # eval_spec 是 tf.estimator.EvalSpec 对象，用于指定验证的输入函数以及其它参数
    
    # max_step 为总训练步数
    # trian_hook 是一个Hook对象, 用来配置分布式训练的参数
    train_spec = TrainSpec(
        input_fn=train_input_fn, max_steps=params.train_steps, hooks=[train_hook])
    
    # throttle_secs 只多少秒后开始新一轮的模型验证
    eval_spec = EvalSpec(
        eval_input_fn, throttle_secs=params.eval_throttle_secs)

    # estimator.train(
    #     train_input_fn, max_steps=params.train_steps, hooks=[train_hook])
    train_and_evaluate(estimator, train_spec, eval_spec)
    return estimator


def eval_bert_multitask(
        problem='weibo_ner',
        num_gpus=1,
        model_dir='',
        eval_scheme='ner',
        params=None,
        problem_type_dict={},
        processing_fn_dict={},
        model=None):
    """Evaluate Multi-task Bert model

    Available eval_scheme:
        ner, cws, acc

    Keyword Arguments:
        problem {str} -- problems to evaluate (default: {'weibo_ner'})
        num_gpus {int} -- number of gpu to use (default: {1})
        model_dir {str} -- model dir (default: {''})
        eval_scheme {str} -- Evaluation scheme (default: {'ner'})
        params {Params} -- params to define model (default: {DynamicBatchSizeParams()})
        problem_type_dict {dict} -- Key: problem name, value: problem type (default: {{}})
        processing_fn_dict {dict} -- Key: problem name, value: problem data preprocessing fn (default: {{}})
    """
    if params is None:
        params = DynamicBatchSizeParams()
    if not params.problem_assigned:

        if model_dir:
            base_dir, dir_name = os.path.split(model_dir)
        else:
            base_dir, dir_name = None, None
        # add new problem to params if problem_type_dict and processing_fn_dict provided
        if processing_fn_dict:
            for new_problem, new_problem_processing_fn in processing_fn_dict.items():
                print('Adding new problem {0}, problem type: {1}'.format(
                    new_problem, problem_type_dict[new_problem]))
                params.add_problem(
                    problem_name=new_problem, problem_type=problem_type_dict[new_problem], processing_fn=new_problem_processing_fn)
        params.assign_problem(problem, gpu=int(num_gpus),
                              base_dir=base_dir, dir_name=dir_name)
        params.from_json()
    else:
        print('Params problem assigned. Problem list: {0}'.format(
            params.problem_list))

    estimator = _create_estimator(
        num_gpus=num_gpus, params=params, model=model)

    evaluate_func = getattr(metrics, eval_scheme+'_evaluate')
    return evaluate_func(problem, estimator, params)


# 预测与评价
def predict_bert_multitask(
        inputs,
        problem='weibo_ner',
        model_dir='',
        params=None,
        problem_type_dict={},
        processing_fn_dict={},
        model=None):

    # inputs  训练数据[[训练集用户msg],[训练集label]]   测试数据[[msg],[label]]
    # new_problem_process_fn_dict 例{'identity1':identity1}
    # new_problem_type 例{'identity1':'cls'}
    """Evaluate Multi-task Bert model

    Available eval_scheme:
        ner, cws, acc

    Keyword Arguments:
        problem {str} -- problems to evaluate (default: {'weibo_ner'})
        num_gpus {int} -- number of gpu to use (default: {1})
        model_dir {str} -- model dir (default: {''})
        eval_scheme {str} -- Evaluation scheme (default: {'ner'})
        params {Params} -- params to define model (default: {DynamicBatchSizeParams()})
        problem_type_dict {dict} -- Key: problem name, value: problem type (default: {{}})
        processing_fn_dict {dict} -- Key: problem name, value: problem data preprocessing fn (default: {{}})
    """

    if params is None:
        params = DynamicBatchSizeParams()
    if not params.problem_assigned:
        if model_dir:
            base_dir, dir_name = os.path.split(model_dir)
        else:
            base_dir, dir_name = None, None
        # add new problem to params if problem_type_dict and processing_fn_dict provided
        # 添加 问题名称， 问题类型，问题数据导入函数（如果有top和share）设置到params中
        if processing_fn_dict:
            for new_problem, new_problem_processing_fn in processing_fn_dict.items():
                print('Adding new problem {0}, problem type: {1}'.format(
                    new_problem, problem_type_dict[new_problem]))
                params.add_problem(
                    problem_name=new_problem, problem_type=problem_type_dict[new_problem], processing_fn=new_problem_processing_fn)
        params.assign_problem(problem, gpu=1,
                              base_dir=base_dir, dir_name=dir_name)
        params.from_json()
    else:
        print('Params problem assigned. Problem list: {0}'.format(
            params.run_problem_list))

    tf.logging.info('Checkpoint dir: %s' % params.ckpt_dir)
    time.sleep(3)

    estimator = _create_estimator(num_gpus=1, params=params, model=model)
    
    # 预测步骤, 直接输入用户文本信息msg, 使用predict_input_fn对象进行
    return estimator.predict(lambda: predict_input_fn(inputs, params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str,
                        default='weibo_ner&weibo_cws', help='Problems to run')
    parser.add_argument('--schedule', type=str,
                        default='train', help='train or eval')
    parser.add_argument('--model_dir', type=str,
                        default='', help='path for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=15)
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()

    if args.schedule == 'train':
        train_bert_multitask(
            problem=args.problem,
            model_dir=args.model_dir,
            num_gpus=args.num_gpus,
            num_epochs=args.num_epochs
        )
    else:
        eval_bert_multitask(
            problem=args.problem,
            model_dir=args.model_dir,
            num_gpus=args.num_gpus,
            eval_scheme=args.eval_scheme,
        )
