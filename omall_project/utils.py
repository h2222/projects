# -*- coding: utf-8 -*-
from urllib.parse import urlparse, urljoin
import re
import os
import csv
import json

punctuation = "~!@#$%^&*()_+`{}|\[\]\:\";\-\\\='<>?,./"
img_postfix = ["svg", "png", "jpg", "jpeg"]

# 消除标点
def remove_punctuation(raw_str, to_blank=False):
    sub_str = ""
    if to_blank is True:
        sub_str = " "
    res = re.sub(r'[{}]+'.format(punctuation), sub_str, raw_str.lower())
    return res


# 基于两个字符串的匹配程度进行评分
#                idx1      idx2
# 例如 str1:  A    B    C    D
#      str2:       B    C    D    E
# score1 = 3, score2 = 0,  final_score = 3
def simple_str_match(str1, str2):
    score1 = 0
    score2 = 0
    str1_len = len(str1)
    str2_len = len(str2)
    if str1_len == 0 or str2_len == 0:
        return 0
    idx1 = str1.index(str2[0]) if str2[0] in str1 else -1
    idx2 = str2.index(str1[0]) if str1[0] in str2 else -1
    if idx1 >= 0:
        for i in range(idx1, str1_len):
            if i < str2_len:
                if str1[i] == str2[i - idx1]:
                    score1 += 1
    if idx2 >= 0:
        for i in range(idx2, str2_len):
            if i < str1_len:
                if str2[i] == str1[i - idx2]:
                    score2 += 1
    return max(score1, score2)


# 数字和字符混合字符串中的字符
def split_digit_str(x):
    for i in range(len(x) - 1):
        if x[i].strip() == "":
            continue
        elif (x[i].isdigit() or (not x[i].isalpha())) and x[i + 1].isalpha():
            return [x[:i + 1], x[i + 1:]]
        elif (x[i] not in [",", "."] and not x[i].isdigit()) and x[i + 1].isdigit():
            return [x[:i + 1], x[i + 1:]]
    return [x]


# 获取纯净的url
def get_clean_url(url):
    """
    获取干净的url链接
    :param
        url: {str} url链接
    :return: {str} 干净的url链接
    """
    ret = urlparse(url)
    if ret.scheme != "":
        link = urljoin(ret.scheme + "://" + ret.netloc, ret.path)
    else:
        link = urljoin(ret.netloc, ret.path)
    return link


# 获取 url 路径
def get_url_path(url):
    ret = urlparse(url)
    return ret.path


# list 消除重复
def remove_repeat_with_order(x):
    res = []
    for e in x:
        if e not in res:
            res.append(e)
    return res


# 批量wget 文件(不一定是img)
def get_img(img_code, url, website="", img_root_path=""):
    urls = [url, urljoin(website, url)]
    if url.startswith("/"):
        urls.append(website + url)
    else:
        urls.append(website + "/" + url)
    for u in urls:
        u = get_clean_url(u)
        end_prefix = u.strip().split(".")[-1]
        try:
            img_path = os.path.join(os.path.abspath(img_root_path), "%s.%s" % (img_code, end_prefix))
            os.system('wget ' + u + '  -O ' + img_path + " --tries 3  --timeout 5")
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                return "success"
            else:
                os.system("rm -rf %s" % img_path)
        except Exception as e:
            print(img_code, u, e)
            continue
    return "fail"


# 移除size == 0 的文件
def remove_zero_size(img_path):
    os.chdir(img_path)
    files = os.listdir(img_path)
    print(len(files))
    for file in files:
        file_size = os.path.getsize(os.path.join(img_path, file))
        if file_size == 0:
            print(file, file_size)
            os.system("rm -rf %s" % os.path.join(img_path, file))


# 文件重命名 
def rename(path):
    os.chdir(img_path)
    files = os.listdir(path)
    for name in files:
        code = name.split(".")[0]
        endfix = name.split(".")[-1]
        print(name, code, endfix)
        os.system("mv %s %s" % (name, "%s.%s" % (code, endfix)))


# 其他格式图片转png(需要安装image magic)
def other2png(img_path):
    os.chdir(img_path)
    files = os.listdir(img_path)
    for name in files:
        if name.endswith("gif"):
            os.system("rm -rf %s" % name)
        elif not name.endswith("png"):
            new_name = ".".join([name.split(".")[0], "png"])
            print(name, new_name)
            os.system("convert %s %s" % (name, new_name))
            os.system("rm -rf %s" % name)


# 压缩文件
def compress(path):
    command_str = """
    pngquant %s -o %s -f --quality 10 --speed 1
    """
    files = os.listdir(path)
    print(len(files))
    for file in files:
        x = ""
        for i in file:
            if i.isdigit():
                x += i
            else:
                break
        old_arr2 = file.split(".")
        new_name = ".".join([x, old_arr2[-1]])
        print(new_name)
        os.system(command_str % (os.path.join(path, file), os.path.join(path, new_name)))


# 刷新文件
def get_fp(f_path):
    if os.path.exists(f_path):
        os.system("rm -rf %s" % f_path)
    return open(f_path, "w")


# dict 转 str 并写入文件
def write_data2file(data, f):
    for key in data:
        f.write(json.dumps({
            key: data[key]
        }) + "\n")
    f.close()
    print("file done")


# dict 转 csv
def write_data2csv(data, f):
    csv_writer = csv.writer(f)
    for key in data:
        val = data[key]
        if isinstance(val, str):
            csv_writer.writerow([key, val])
        elif isinstance(val, list):
            csv_writer.writerow([key] + val)
    print("csv done")


# f 转 list
def load_raw_text(f_path):
    arr = []
    for line in open(f_path):
        arr.append(line.strip())
    return arr


# 中文括号处理
def sh_escape(s):
    return s.replace("(", "\\(").replace(")", "\\)")


# list 转 txt
def data2file(data, f_path):
    if len(data) == 0:
        return
    if os.path.exists(f_path):
        return
    f_new = open(f_path, "w", encoding="utf-8")
    for line in data:
        f_new.write(line)
    f_new.close()


# list 转 csv
def data2csv(data, f_path):
    if len(data) == 0:
        return
    if os.path.exists(f_path):
        return
    f_new = open(f_path, "w", encoding="utf-8")
    csv_write = csv.writer(f_new)
    for d in data:
        csv_write.writerow(d)
    f_new.close()


# 没暖用的字符串判断
def no_meaning_elem(x, th=3):
    raw_dict = {}
    for i in x:
        if i not in raw_dict:
            raw_dict[i] = 0
        raw_dict[i] += 1
    if set(raw_dict.keys()).issubset(set([" ", "", "-", "_", "."])):
        if max(raw_dict.values()) > th:
            return True
    return False


# 取消连接符
def cut_mark(x, mark="-"):
    middle_line_num = x.count(mark)
    if middle_line_num >= 3:
        x = x.strip(mark)
        return x
    return x

# just for fun
def grep_file(f_path, elem):
    try:
        line = os.popen("cat %s | grep '%s'" % (f_path, elem)).read()[:100]
    except Exception as e:
        print(e)
        pass
    return line