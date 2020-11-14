# -*- coding: utf-8 -*-
from ufile import filemanager, bucketmanager
from ufile import config
import time
import os
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
import smtplib
import MySQLdb
import json
# config.set_default(uploadsuffix='.internal-afr-nigeria.ufileos.com')
# # 设置请求连接超时时间，单位为秒
# config.set_default(connection_timeout=60)
# # 设置私有bucket下载链接有效期,单位为秒
# config.set_default(expires=60)
# public_key = 'TOKEN_9ccff93a-c13c-41ca-bffc-d85020efc4be'
# private_key = 'af6a6c89-4ebd-4708-ade3-cd6deccdbb6a'
# public_bucket = 'opay-datalake'  # 公共空间名称

sender = 'bowen.wang@opay-inc.com'
password = 'Wbwgh1215@'

DB_ADDR = "rm-d7o3zz06joqq70d2872520.mysql.eu-west-1.rds.aliyuncs.com"
USER = "algouser"
PINCODE = "WOPnwxW5"
DB = "algorithm"
PORT = 3306

ALIYUN_DB_ADDR = "rm-d7o3zz06joqq70d2872520.mysql.eu-west-1.rds.aliyuncs.com"
ALIYUN_USER = "algouser"
ALIYUN_PINCODE = "WOPnwxW5"
ALIYUN_DB = "algorithm"
ALIYUN_PORT = 3306

TEST_DB_ADDR = "10.52.128.191"
TEST_USER = "algo"
TEST_PINCODE = "fJv*8#F1.u"
TEST_DB = "algorithm"

ORIDE_DB_ADDR = "rm-d7obq57mm4k5geuzd72530.mysql.eu-west-1.rds.aliyuncs.com"
ORIDE_USER = "oride"
ORIDE_PINCODE = "c8cebfe76683"
ORIDE_DB = "oride_data"
ORIDE_PORT = 3306

ORIDE_TEST_DB_ADDR = "152.32.140.126"
ORIDE_TEST_USER = "oride"
ORIDE_TEST_PINCODE = "1q2ws3edcRFVTGB"
ORIDE_TEST_DB = "oride_data"

ORIDE_TEST_OEXP_DB_ADDR = "152.32.140.126"
ORIDE_TEST_OEXP_USER = "oexpress"
ORIDE_TEST_OEXP_PINCODE = "1q2ws3edcRFVTGB"
ORIDE_TEST_OEXP_DB = "oexpress_data"

OEXP_DB_ADDR = "rm-d7oa858f8lju41w5n3o.mysql.eu-west-1.rds.aliyuncs.com"
OEXP_USER = "expressuser"
OEXP_PINCODE = "aRaY2UEkXUZ95z27"
OEXP_DB = "oexpress_data"

OEXP_READONLY_DB_ADDR = "rr-d7ot8d41596u8edx4oo.mysql.eu-west-1.rds.aliyuncs.com"
OEXP_READONLY_USER = "read_only"
OEXP_READONLY_PINCODE = "y^n#^qk3"
OEXP_READONLY_DB = "oexpress_data"

OMALL_READONLY_DB_ADDR = "10.52.28.144"
OMALL_READONLY_USER = "read_only"
OMALL_READONLY_PINCODE = "y^n#^qk3"
OMALL_READONLY_DB = "opay_mall_platform"

OMALL_TEST_DB_ADDR = "47.244.61.60"
OMALL_TEST_USER = "root"
OMALL_TEST_PINCODE = "Oloan@123"
OMALL_DB = "opay_mall_platform"
OMALL_STAT_DB = "opay_mall_stat"

OMALL_DB_ADDR = "rm-d7oxg2z29xe5wc8slxo.mysql.eu-west-1.rds.aliyuncs.com"
OMALL_USER = "omall"
OMALL_PINCODE = "CGD+2qD+D5DCoiTrmZp3H0EY0Ag="

STAT_ADDR = "rm-d7o35sbc80l4o5lt5.mysql.eu-west-1.rds.aliyuncs.com"
STAT_USER = "root"
STAT_PINCODE = "Q1BoLf8aHxNGaRxS"
STAT_DB = "oride_data_restore"

oexp_receivers = [
    "jun.ma@opay-inc.com",
    "shuai.li@opay-inc.com",
    "chaoqun.wang@opay-inc.com",
    "tao.zhang@opay-inc.com",
    "tieping.sun@opay-inc.com",
    "jianwei.qiao@opay-inc.com",
    "kaijian.zhao@opay-inc.com",
    "oride-algorithm@opay-inc.com",
]

def send_email(title, receivers, msg, attach_paths):
    message = MIMEMultipart()
    subject = title
    message['Subject'] = Header(subject, 'utf-8')
    message.attach(MIMEText(msg, 'html', 'utf-8'))
    for p in attach_paths:
        if os.path.exists(p):
            att = MIMEText(open(p, 'r').read(), 'plain', 'utf-8')
            att["Content-Type"] = 'application/octet-stream'
            att["Content-Disposition"] = 'attachment; filename="%s"' % p
            message.attach(att)
    try:
        server = smtplib.SMTP_SSL('mail.opay-inc.com', 465)
        # server.ehlo()
        # server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e.message)

def send_email_v2(title, receivers, msg, attach_paths):
    message = MIMEMultipart()
    subject = title
    message['Subject'] = Header(subject, 'utf-8')
    message.attach(MIMEText(msg, 'html', 'utf-8'))
    for p in attach_paths:
        if os.path.exists(p):
            att = MIMEText(open(p, 'r').read(), 'plain', 'utf-8')
            att["Content-Type"] = 'application/octet-stream'
            att["Content-Disposition"] = 'attachment; filename="%s"' % p
            message.attach(att)
    try:
        server = smtplib.SMTP_SSL('mail.opay-inc.com', 465)
        # server.ehlo()
        # server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e.message)

# def check_file(remote_path):
#     res = False
#     postufile_handler = filemanager.FileManager(public_key, private_key)
#     ret, resp = postufile_handler.getfilelist(public_bucket, remote_path)
#     if resp.status_code == 200:
#         for elem in ret["DataSet"]:
#             if "_SUCCESS" in elem['FileName']:
#                 res = True
#                 print("success check: " + remote_path)
#                 break
#     return res

def get_dts(dt):
    timeArray = time.strptime(dt, "%Y-%m-%d")
    timestamp = int(time.mktime(timeArray))
    dts = []
    for t in range(1, 10):
        tmp_dt = time.strftime('%Y-%m-%d', time.localtime(timestamp - 86400 * t))
        dts.append(tmp_dt)
    dt1 = time.strftime('%Y-%m-%d', time.localtime(timestamp - 86400))
    return dts, dt1

def get_db_cursor(env):
    if env == "test-algo":
        db_addr, user, pincode, db, port = TEST_DB_ADDR, TEST_USER, TEST_PINCODE, TEST_DB, PORT
    elif env == "test-oride":
        db_addr, user, pincode, db, port = ORIDE_TEST_DB_ADDR, ORIDE_TEST_USER, ORIDE_TEST_PINCODE, ORIDE_TEST_DB, PORT
    elif env == "test-oexp":
        db_addr, user, pincode, db, port = ORIDE_TEST_OEXP_DB_ADDR, ORIDE_TEST_OEXP_USER, ORIDE_TEST_OEXP_PINCODE, ORIDE_TEST_OEXP_DB, PORT
    elif env == "prod":
        db_addr, user, pincode, db, port = DB_ADDR, USER, PINCODE, DB, PORT
    elif env == "oexp":
        db_addr, user, pincode, db, port = OEXP_DB_ADDR, OEXP_USER, OEXP_PINCODE, OEXP_DB, PORT
    elif env == "oexp-readonly":
        db_addr, user, pincode, db, port = OEXP_READONLY_DB_ADDR, OEXP_READONLY_USER, OEXP_READONLY_PINCODE, OEXP_READONLY_DB, PORT
    elif env == "omall-readonly":
        db_addr, user, pincode, db, port = OMALL_READONLY_DB_ADDR, OMALL_READONLY_USER, OMALL_READONLY_PINCODE, OMALL_READONLY_DB, PORT
    elif env == "test-omall":
        db_addr, user, pincode, db, port = OMALL_TEST_DB_ADDR, OMALL_TEST_USER, OMALL_TEST_PINCODE, OMALL_DB, PORT
    elif env == "test-stat-omall":
        db_addr, user, pincode, db, port = OMALL_TEST_DB_ADDR, OMALL_TEST_USER, OMALL_TEST_PINCODE, OMALL_STAT_DB, PORT
    elif env == "omall":
        db_addr, user, pincode, db, port = OMALL_DB_ADDR, OMALL_USER, OMALL_PINCODE, OMALL_DB, PORT
    elif env == "stat-omall":
        db_addr, user, pincode, db, port = OMALL_DB_ADDR, OMALL_USER, OMALL_PINCODE, OMALL_STAT_DB, PORT
    elif env == "stat":
        db_addr, user, pincode, db, port = STAT_ADDR, STAT_USER, STAT_PINCODE, STAT_DB, PORT
    else:
        db_addr, user, pincode, db, port = ALIYUN_DB_ADDR, ALIYUN_USER, ALIYUN_PINCODE, ALIYUN_DB, ALIYUN_PORT
    db = MySQLdb.connect(host=db_addr,user=user, passwd=pincode, db=db,port=port, charset='utf8mb4')
    return db

def insert_sql(sql,res,env):
    db = get_db_cursor(env)
    cursor = db.cursor()
    cursor.executemany(sql,res)
    db.commit()
    cursor.close()

def insert_sql_by_one(sql,res,env):
    db = get_db_cursor(env)
    cursor = db.cursor()
    cursor.executemany(sql,res)
    db.commit()
    cursor.close()

def del_sql(sql,env):
    db = get_db_cursor(env)
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    cursor.close()

def query_sql(sql,env):
    db = get_db_cursor(env)
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    res = cursor.fetchone()
    cursor.close()
    return res[0]

def query_sql_all(sql,env):
    db = get_db_cursor(env)
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    res = cursor.fetchall()
    cursor.close()
    return res

def get_oride_db_cursor(env):
    if env == "test":
        db_addr, user, pincode, db, port = ORIDE_TEST_DB_ADDR, ORIDE_TEST_USER, ORIDE_TEST_PINCODE, ORIDE_TEST_DB, PORT
    else:
        db_addr, user, pincode, db, port = ORIDE_DB_ADDR, ORIDE_USER, ORIDE_PINCODE, ORIDE_DB, ORIDE_PORT
    db = MySQLdb.connect(host=db_addr,user=user, passwd=pincode, db=db,port=port, charset='utf8mb4')
    return db

def oride_query_sql(sql,env):
    db = get_oride_db_cursor(env)
    cursor = db.cursor()
    cursor.execute(sql)
    db.commit()
    res = cursor.fetchall()
    cursor.close()
    return res

def update_by_local_file(local_file_path,insert_query,column_list,column_dict,extra_column_list,env):
    os.chdir(local_file_path)
    # 获取你路径下的所有文件名
    file_list = os.listdir(local_file_path)
    for file_name in file_list:
        res = []
        count = 0
        with open(file_name, "r") as f:
            lines = f.readlines()  # 读取全部内容
            for line in lines:
                count += 1
                insert_data_list = []
                data = json.loads(line)
                for column in column_list:
                    insert_data = data[column]
                    if insert_data is None:
                        insert_data = column_dict[column]
                    insert_data_list.append(insert_data)
                insert_data_list.extend(extra_column_list)
                res.append(insert_data_list)
                if count % 2000 == 0:
                    insert_sql(insert_query, res, env)
                    res = []

        insert_sql(insert_query, res, env)