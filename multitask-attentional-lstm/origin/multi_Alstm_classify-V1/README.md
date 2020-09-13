多任务lstm文件

run_att_bilstm.sh是运行训练脚本

run_predict.sh是运行预测脚本

models/attn_bi_lstm.py是训练文件

models/predict.py是预测文件

models/Config/config.cfg是配置文件

models/utils/data_helper.py是数据加载文件

models/utils/model_helper.py是执行函数文件

ckpt2pb.py是转pb模型文件

执行需要修改的相关信息在config.cfg文件中，log文件在当日日期文件夹下，训练和预测需要更改config中相关参数

训练或预测必须要修改config.cfg中的以下信息：

    data_dir：数据存放路径
              训练验证测试数据文件分别是：${cls_type}_train.csv、${cls_type}_test.csv、${cls_type}_predict.csv 
              其中cls_type为具体类别，for example：ask_know_train.csv、ask_know_test.csv、ask_know_predict.csv
              每个文件需要有use_cols=['type_robot','msg','type','type_combine']列，for example：['询问是否认识借款人','我认识','肯定','yes']
            
    save_dir：模型保存路径
    
    is_train = True ：True为训练，False为预测
    
    path_model_Predict：预测模型的路径（保存模型的路径）

run_att_bilstm.sh 运行训练脚本中的python自行修改
run_predict.sh 运行预测脚本中的python自行修改

ckpt2pb.py 转pb模型文件 需要修改模型路径，根据训练模型中的类别（cls_type）信息修改需要预测的out_put