#!/usr/bin/env bash
. /etc/profile
. ~/.bash_profile
cd /home/hdfs/bowen.wang/algo-offline-job/omall/omall_stat
PYTHONPATH=./ /usr/lib/spark-current/bin/spark-submit --master yarn-client --queue algo_spark ./omall_hourly_stat.py prod >> /tmp/omall_hourly_stat.log
PYTHONPATH=./ /usr/lib/spark-current/bin/spark-submit --master yarn-client --queue algo_spark ./omall_hourly_stat.py test