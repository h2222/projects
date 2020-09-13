cur_dateTime=`date +%m%d-%H%M`
export LC_CTYPE=en_US.UTF-8
mkdir ./out_put/${cur_dateTime} #创建log.txt存储文件路径
nohup /opt/anaconda3-5.1.0/bin/python3.6 ./models/attn_bi_lstm.py > ./out_put/${cur_dateTime}/log.txt 2>&1 &

