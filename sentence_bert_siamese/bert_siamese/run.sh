cur_dateTime=`date +%m%d-%H%M`
cp $0 codes/run_${cur_dateTime}.sh
python_name='run_siamese'
cp ${python_name}.py codes/${python_name}_${cur_dateTime}.py
#nohup /opt/anaconda3-5.1.0/bin/python3.6 ${python_name}.py --data_dir=/home/zhaoxi.li/bert/data --task_name=mine --vocab_file=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/bert_model.ckpt --output_dir=./out_put/${cur_dateTime} --num_train_epochs=1 --do_train=True --gpu=2 --raw_data=False --train_data=train_bert_87_triplet_cut20+invalid+label.csv --do_predict=True --predict_data=test_bert_87_triplet_cut20.csv --predict_output=test_result.tsv.1 >logs/log_${cur_dateTime}.txt 2>&1 &
# nohup /opt/anaconda3-5.1.0/bin/python3.6 ${python_name}.py --data_dir=/home/zhaoxi.li/bert/data --task_name=mine --vocab_file=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/bert_model.ckpt --output_dir=./out_put/${cur_dateTime} --num_train_epochs=1 --do_train=True --gpu=2 --raw_data=True --train_data=train_bert_mean_1_85_tokenize_add1.csv --do_predict=True --predict_data=test_bert_mean_1_85_tokenize_add1.csv --predict_output=test_result.tsv.1 >logs/log_${cur_dateTime}.txt 2>&1 &
nohup /opt/anaconda3-5.1.0/bin/python3.6 ${python_name}.py --data_dir=/home/zhaoxi.li/bert/data_no_repeat --task_name=mine --vocab_file=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=/home/zhaoxi.li/bert/model_save/chinese_L-12_H-768_A-12/bert_model.ckpt --output_dir=./out_put/${cur_dateTime} --num_train_epochs=1 --do_train=True --gpu=2 --raw_data=True --train_data=train.csv --do_predict=True --predict_data=test.csv --predict_output=test_result.tsv.1 >logs/log_${cur_dateTime}.txt 2>&1 &