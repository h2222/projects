# -*- coding: utf-8 -*-
import sys
import time
import json
from init_spark import init_spark
import csv
from aliyun_util import *
from datetime import datetime
import math

hourly_stat_query_sql = '''
select
merchant_id,
'{last_dt}' dt,
sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') = '{last_dt}' and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,all_price,0)) yes_sale_num,
sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') = '{last_dt}' and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,1,0)) yes_order_num,
sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%u') = from_unixtime(unix_timestamp('{last_dt}'),'%Y-%u') and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,all_price,0)) week_sale_num,
sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%u') = from_unixtime(unix_timestamp('{last_dt}'),'%Y-%u') and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,1,0)) week_order_num,
sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m') = from_unixtime(unix_timestamp('{last_dt}'),'%Y-%m') and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,all_price,0)) month_sale_num,
sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m') = from_unixtime(unix_timestamp('{last_dt}'),'%Y-%m') and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,1,0)) month_order_num,
sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') >= '{last_7_dt}' and from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') <= '{last_dt}' and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,all_price,0)) last_7D_sale_num
from nideshop_order
group by merchant_id,'{last_dt}'
'''

insert_current_stat_sql = '''
insert into data_omall_dashboard_history_info (`merchant_id`,`dt`,`yes_sale_num`,`yes_order_num`,
`week_sale_num`,`week_order_num`,`month_sale_num`,`month_order_num`,`last_7D_sale_num`)
values (%s,%s,%s,%s,%s,%s,%s,%s,%s) on duplicate key update yes_sale_num=values(yes_sale_num),
yes_order_num=values(yes_order_num),week_sale_num=values(week_sale_num),week_order_num=values(week_order_num),
month_sale_num=values(month_sale_num),month_order_num=values(month_order_num),last_7D_sale_num=values(last_7D_sale_num)
'''

def insert_daily_stat_data():
    res = []
    if env == 'test':
        query_env = env + "-omall"
        stat_env = env + "-stat-omall"
    else:
        query_env = "omall"
        stat_env = "stat-omall"
    results = query_sql_all(hourly_stat_query_sql.format(last_dt=last_dt,last_7_dt=last_7_dt),query_env)
    for data in results:
        res.append(list(data))
    insert_sql(insert_current_stat_sql,res,stat_env)
    return res

if __name__ == "__main__":
    program_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    spark, sc = init_spark('omall_history_stat')
    sc.setLogLevel("WARN")
    env = str(sys.argv[1])
    last_dt = str(sys.argv[2]) if len(sys.argv) > 2 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400))
    last_7_dt = str(sys.argv[2]) if len(sys.argv) > 2 else time.strftime('%Y-%m-%d', time.localtime(time.time() - 86400*7))
    # dt = '2020-06-02'
    res = insert_daily_stat_data()
