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
t1.merchant_id,
from_unixtime(unix_timestamp(),'%Y-%m-%d') dt,
if(from_unixtime(unix_timestamp(),'%H')=23,0,from_unixtime(unix_timestamp(),'%H')+1) h,
coalesce(pending_payment_num,0) pending_payment_num,
coalesce(pending_dispatch_num,0) pending_dispatch_num,
coalesce(in_transit_num,0) in_transit_num,
coalesce(cancel_num,0) cancel_num,
coalesce(today_order_num,0) today_order_num,
coalesce(today_sale_num,0) today_sale_num,
off_shelf_product_num,
on_shelf_product_num,
on_warehouse_product_num,
to_audit_product_num,
all_order_num,
coalesce(today_hourly_order_num,0) today_hourly_order_num,
coalesce(today_hourly_sale_num,0) today_hourly_sale_num
from
(
  select 
  merchant_id,
  sum(if(is_on_sale=0,1,0)) off_shelf_product_num,
  sum(if(is_on_sale=1,1,0)) on_shelf_product_num,
  sum(if(is_on_sale=2,1,0)) on_warehouse_product_num,
  count(*) as all_order_num,
  sum(if(audit_status=1,1,0)) to_audit_product_num
  from nideshop_goods
  group by merchant_id
) t1
left join
(
  select
  merchant_id,
  sum(if(pay_status=0 and order_status not in (101,103),1,0)) pending_payment_num,
  sum(if(order_status=201,1,0)) pending_dispatch_num,
  sum(if(order_status=300,1,0)) in_transit_num,
  sum(if(order_status=101,1,0)) cancel_num,
  sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') = '{dt}' and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,1,0)) today_order_num,
  sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') = '{dt}' and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,all_price,0)) today_sale_num,
  sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') = '{dt}' and from_unixtime(UNIX_TIMESTAMP(pay_time),'%H')  = from_unixtime(unix_timestamp(),'%H') and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,1,0)) today_hourly_order_num,
  sum(if(from_unixtime(UNIX_TIMESTAMP(pay_time),'%Y-%m-%d') = '{dt}' and from_unixtime(UNIX_TIMESTAMP(pay_time),'%H')  = from_unixtime(unix_timestamp(),'%H') and order_status in (201,300,301) and shipping_status != 4 and pay_status != 4,all_price,0)) today_hourly_sale_num
  from nideshop_order
  group by merchant_id
) t2
on t1.merchant_id=t2.merchant_id 
'''

insert_hourly_stat_sql = '''
insert into data_omall_dashboard_current_info (`merchant_id`,`dt`,`h`,`pending_payment_num`,`pending_dispatch_num`,
`in_transit_num`,`cancel_num`,`today_order_num`,`today_sale_num`,`off_shelf_product_num`,`on_shelf_product_num`,
`on_warehouse_product_num`,`to_audit_product_num`,`all_order_num`,`today_hourly_order_num`,`today_hourly_sale_num`)
values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) on duplicate key update pending_payment_num=values(pending_payment_num),
pending_dispatch_num=values(pending_dispatch_num),in_transit_num=values(in_transit_num),cancel_num=values(cancel_num),
today_order_num=values(today_order_num),today_sale_num=values(today_sale_num),off_shelf_product_num=values(off_shelf_product_num),
on_shelf_product_num=values(on_shelf_product_num),on_warehouse_product_num=values(on_warehouse_product_num),
to_audit_product_num=values(to_audit_product_num),all_order_num=values(all_order_num),today_hourly_order_num=values(today_hourly_order_num),
today_hourly_sale_num=values(today_hourly_sale_num)
'''

def insert_hourly_stat_data():
    res = []
    if env == 'test':
        query_env = env + "-omall"
        stat_env = env + "-stat-omall"
    else:
        query_env = "omall"
        stat_env = "stat-omall"
    results = query_sql_all(hourly_stat_query_sql.format(dt=dt),query_env)
    for data in results:
        res.append(list(data))
    insert_sql(insert_hourly_stat_sql,res,stat_env)
    return res

if __name__ == "__main__":
    program_start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    spark, sc = init_spark('omall_hourly_stat')
    sc.setLogLevel("WARN")
    env = str(sys.argv[1])
    dt = str(sys.argv[2]) if len(sys.argv) > 2 else time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # dt = '2020-06-02'
    res = insert_hourly_stat_data()
