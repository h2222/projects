# -*- coding: utf-8 -*-
import oss2
import os
from consts import *
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
sender = 'research@opay-inc.com'
password = 'G%4nlD$YCJol@Op'

receivers = ['lichang.zhang@opay-inc.com', 'chengyang.wang@opay-inc.com', 'cancan.ma@opay-inc.com']
min_action = 2


def get_bucket():
    auth = oss2.Auth('LTAI4FmshDAuao6Wyn1BAoGB', 'MLhLZpdpL1sNrtUNsfcfcFj4eHq7kp')
    bucket = oss2.Bucket(auth, 'http://oss-eu-west-1-internal.aliyuncs.com', 'oride-algo')
    return bucket


def upload_via_oss(fp, oss_path):
    bucket = get_bucket()
    with open(fp, 'rb') as fileobj:
        bucket.put_object(oss_path, fileobj)
    # os.system("rm -rf %s" % fp)


def get_fp(f_path):
    if os.path.exists(f_path):
        os.system("rm -rf %s" % f_path)
    return open(f_path, "aw")


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
        server.login(sender, password)
        server.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(e.message)