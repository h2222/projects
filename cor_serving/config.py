#coding=utf-8


# http boddy 格式
boddy = {'word':'货款多少',     # 用户文本
         'id':'1001',           # 百应系统话术id
         'currentNodeId':'1.1',  # 当前对话子节点id
         'status':0,            # 状态码, 0整除, 其他:错误
         'answer':'default',    # 分析得出的用户语义
         'score':0              # 得分
        }


# 预训练模型路径(用于分类模型的预处理)
prem_path = '/home/jiaxiang.hao/NLP/pre_model_and_vocob/chinese_L-12_H-768_A-12'
vocab_path = '/home/jiaxiang.hao/NLP/pre_model_and_vocob/vocab.txt'


### 排序模型
# url
url_sort = 'http://192.168.23.90:8088/score/v1/get_data'
# 服务调用json
params_sort = {"appName":"cfs","modelName":"semantic_analysis_new","version":"V0_25","bizParam":{"model":"predict","inparam":{"in_node":"6468","msg":"default","process_id":"1404"},"outLevel":2,"callid":"test_001","caseid":"test_001","in_way_id":"test_001","voice_status":"voice_play_now"},"traceId":"1233434"}
# 027 版本排序模型, 输入processid 为 benrenshoucui
params_sort_027 = {"appName":"cfs","modelName":"semantic_analysis_new","version":"V0_27","bizParam":{"model":"predict","inparam":{"in_node":"1.1","msg":"default","processid":"benrenshoucui"},"outLevel":2,"callid":"test_001","caseid":"test_001","in_way_id":"test_001","voice_status":"voice_play_now"},"traceId":"1233434"}


# 排序模型,出节点与对应意图的映射关系
#h_sort = {'1.1':'身份确认', '2.1':'本人', '2.2':'非本人', '3.2':'账单日前能处理', '3.4':'账单日前不能处理', '3.5':'关联人', '3.6':'非关联人', '敏感词':'敏感词', '知识库':'知识库', '-99':'~', '':'无效语义', 'm_paid':'敏感词', 'z_askcompany':'知识库', 'z_asktime':'知识库', 'z_askmoney':'知识库', '6040':'AI_UNKNOWN', '6041':'BREAK', '6042':'AI_UNKNOWN_END', '6043':'USER_NOT_ANSWER', '6044':'NULL', '6045':'OK', '6046':'REJECT', '-97':'~'}
h_sort = {'6468':'身份确认', '6469':'OK', '6470':'亲友+REJECT+~', '6473':'OK', '6472':'REJECT+~', '6475':'亲友+OK', '6476':'本人', '6477':'REJECT+~', '敏感词':'敏感词', '知识库':'知识库', '-99':'~', '':'~', 'm_paid':'敏感词', 'z_askcompany':'知识库', 'z_asktime':'知识库', 'z_askmoney':'知识库', '6040':'AI_UNKNOWN', '6041':'BREAK', '6042':'AI_UNKNOWN_END', '6043':'USER_NOT_ANSWER', '6044':'NULL', '6045':'OK', '6046':'REJECT', '-97':'~'}
# in_node 类型(baiying)
# in_node_sort = ['6040', '6041', '6042', '6043', '6044', '6045', '6046', '6468', '6469', '6470', '6471', '6473', '6472', '6474', '6475', '6477']


### 分类模型
# url
url_cls = 'http://192.168.23.90:7631/v1/models/tts:predict'
# 服务调用json
params_cls =  {"instances":"default"}
# headers
header_cls = {"content-type":"application/json"}
# 分类模型,出节点与对应意图的映射关系
#h_cls = {'1.1':'身份确认', '2.1':'本人', '2.2':'非本人', '3.1':'账单日前能处理', '3.2':'账单日前不能处理', '3.3':'关联人', '3.4':'非关联人', '敏感词':'敏感词', '知识库':'知识库', '-99':'~', '':'~'}
h_cls = {'6468':'身份确认', '6469':'OK', '6470':'亲友+REJECT+~', '6473':'OK', '6472':'REJECT+~', '6475':'亲友+OK', '6476':'本人', '6477':'REJECT+~', '敏感词':'敏感词', '知识库':'知识库', '-99':'~', '':'~', '6040':'AI_UNKNOWN', '6041':'BREAK', '6042':'AI_UNKNOWN_END', '6043':'USER_NOT_ANSWER', '6044':'NULL', '6045':'OK', '6046':'REJECT', '-97':'~'}




