#!/usr/bin/env bash
. /etc/profile
. ~/.bash_profile
cd /home/hdfs/bowen.wang/algo-offline-job/omall/omall_stat
PYTHONPATH=./ /usr/lib/spark-current/bin/spark-submit --master yarn-client --queue algo_spark ./omall_history_stat.py prod
PYTHONPATH=./ /usr/lib/spark-current/bin/spark-submit --master yarn-client --queue algo_spark ./omall_history_stat.py test