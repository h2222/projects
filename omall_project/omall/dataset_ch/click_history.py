# -*- coding: utf-8 -*-
import os
import json
import sys
import time
from datetime import datetime, timedelta
from pyspark.sql import SparkSession


def init_spark(app_name):
    spark = SparkSession \
        .builder \
        .appName(app_name) \
        .enableHiveSupport() \
        .getOrCreate()
    sc = spark.sparkContext
    return spark, sc


def mapper1(x):
    try:
        tmp = json.loads(x)
        label = int(tmp["label"])
        if label == 0:
            return
        uid = int(tmp["user_id"])
        gid = int(tmp["good_id"])
        ts = int(tmp["ts"])
        return uid, (gid, ts)
    except:
        pass


def sort_d(x):
    x = list(x)
    value = x[1]
    value.sort(key=lambda x: x[1])
    x[1] = value
    return tuple(x)


def get_dataset(start_dt, days=7):
    data = sc.parallelize([])
    dt = datetime.strptime(start_dt, '%Y-%m-%d')
    for i in range(days):
        try:
            cur_dt = dt - timedelta(days=i)
            ret = sc.textFile(
                "oss://opay-datalake/algo_migration/omall/event_label/dt=%s/*" % cur_dt.strftime('%Y-%m-%d'))
            data = data.union(ret)
            print(cur_dt)
        except:
            pass
    data.map(mapper1)\
        .filter(lambda x: x is not None)\
        .groupByKey()\
        .mapValues(list)\
        .map(sort_d)\
        .filter(lambda x: x is not None)\
        .map(lambda (uid, click_history): json.dumps({"user_id": uid, "click_history": click_history}))\
        .filter(lambda x: x is not None)\
        .repartition(1)\
        .saveAsTextFile("/tmp/research/mall_%s_%s_%d/" %("click_history", start_dt, days),
                         compressionCodecClass="org.apache.hadoop.io.compress.GzipCodec")


def download_unzip_data(dt, d, root_path, type_name="click_history"):
    root_path += type_name
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    tmp_dataset_path = "/tmp/research/mall_%s_%s_%d/" % (type_name, dt, d)
    dataset_path = "{}/{}/".format(root_path, 'mall_%s_%s_%d'%(type_name, dt, d))
    os.system("/usr/bin/sh /data/download.sh %s %s" % (tmp_dataset_path, root_path))
    os.system("/usr/bin/gunzip -d {}{}".format(dataset_path, "part-00000.gz"))
    return dataset_path+"/part-00000", root_path


def to_text(fname, fdir):
    # assume user/item index starting from 1
    f = open(fname, 'r')
    fs = open(fdir+'/record_file/click_history.txt', 'w')
    for line in f.readlines():
        d = eval(line)
        d = eval(line)
        u = d['user_id']
        ch = d['click_history']
        if len(ch) >= 4:
            with open(fdir+'/record_file/click_history.txt', 'a') as fs:
                for i in ch:
                    fs.write(str(u)+' '+str(i[0])+'\t\n')     



if __name__ == "__main__":
    spark, sc = init_spark("click_history")
    sc.setLogLevel("ERROR")
    os.system('hdfs dfs -rm -r /tmp/research/*click_history*')
    days = int(sys.argv[1]) if len(sys.argv)>1 else 28
    train_dt = str(sys.argv[2]) if len(sys.argv)>2 else time.strftime('%Y-%m-%d',time.localtime(time.time() - 86400 * (1 + days)))    
    get_dataset(train_dt, days) 
    fpath, fdir = download_unzip_data(train_dt, days, '/data/')
    print('dataset file unzipped')
    to_text(fpath, fdir)
    print('build text file for train and eval')
