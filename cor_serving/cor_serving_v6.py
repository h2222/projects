import os, sys
sys.path.append('/home/jiaxiang.hao/NLP/multi_bert_record_win')
from bert_multitask_learning import (to_serving_input, FullTokenizer, PREDICT, DynamicBatchSizeParams)
import numpy as np
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import aiohttp, asyncio
from config import boddy, prem_path, vocab_path, url_sort, params_sort, h_sort,in_node_sort, url_cls, params_cls, header_cls, h_cls


app = Flask(__name__)
CORS(app,  resources={r"/*": {"origins": "*"}})

params = DynamicBatchSizeParams()
params.init_checkpoint = prem_path

async def models(url, data, headers):
    async with aiohttp.ClientSession() as session:
        # 8088 为排序模型端口, url中包含8088即为排序模型
        if '8088' in url:
            async with session.post(url, json=data) as resp:
                return await resp.json()

        # 分类模型
        else:
            async with session.post(url, data=data, headers=headers) as resp:
                return await resp.json()

#class JSONResponse(Response):
#    """
#    返回数据封装
#    """
#    default_mimetype = 'application/json'
#
#    def __init__(self, msg='', in_node='6044', processid='1393', **kwargs):
#        
#        boddy['word'] = msg
#        boddy['id'] = processid
#        boddy['currentNodeId'] = in_node
#
#        super(JSONResponse, self).__init__(json.dumps(body), **kwargs)


@app.route('/async_2models',methods=['POST'])
def test():
    '''
    每个模型是一个协程
    异步请求, 一个线程内塞两个协程
    '''
    if request.method=='POST':

        # json 中拿取数据, 导入至2和模型的json请求中
        try:
            json_data = request.get_json(force=True)
        except Exception as e:
            boddy['status'] = 1
            boddy['word'] = 'the input json is wrong'
            return jsonify(boddy)

        # 排序模型, 获取word, currentNodeId, id
        params_sort['bizParam']['inparam']['msg'] = json_data['word']
        params_sort['bizParam']['inparam']['in_node'] = json_data['currentNodeId']
        params_sort['bizParam']['inparam']['process_id'] = json_data['id']


        # 分类模型, 获取word
        x = next(to_serving_input([json_data['word']], params, PREDICT, FullTokenizer(vocab_path)))
        params_cls['instances'] = [x]
        param_cls = json.dumps(params_cls)

        
        # 启动时间循环(一个线程)
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        # 建立事件循环
        tasks = [models(url_sort, params_sort, None), models(url_cls, param_cls, header_cls)]
        # 执行事件循环并返回结果
        result = event_loop.run_until_complete(asyncio.wait(tasks))
        dataset = [x.result() for x in list(result[0])]
        event_loop.close()      
       


        # 因为异步, 判断dataset前后谁是sort谁是cls
        if 'predictions' in dataset[0]:
            dataset = {'cls_result':dataset[0], 'sort_result':dataset[1]}
        else:
            dataset = {'cls_result':dataset[1], 'sort_result':dataset[0]}

        

        final_result = {}
        sim = 0.0
        prob = 0.0
        prodiction = None
        out_nodes = {}
        label2id = {0:'invalid', 1:'invalid', 2:'yes', 3:'no'}



        # 判断相似度是否大于阈值
        try:
            sim = float(dataset['sort_result']['resultData']['semantic_analysis_new_V0_25']['similarity'])
        except:
            # 命中规则, sim等于1
            sim = 1
        if sim >= 0.8 or json_data['currentNodeId'] in in_node_sort:
            outnode_sort = dataset['sort_result']['resultData']['semantic_analysis_new_V0_25']['out_node']
            outype_sort = h_sort[str(outnode_sort)]
            
            final_result = {'status':0, 'answer':outype_sort, 'score':sim}
        else:

            if json_data['currentNodeId'] == '1.1':
                prediction = dataset['cls_result']['predictions'][0]['identity1']
                out_nodes = {'yes':'2.1', 'no':'2.2', 'invalid':'-99'}
            elif json_data['currentNodeId'] == '2.1':
                prediction = dataset['cls_result']['predictions'][0]['dealml']
                out_nodes = {'yes':'3.1', 'no':'3.2', 'invalid':'-99'}
            elif json_data['currentNodeId'] == '2.2':
                prediction = dataset['cls_result']['predictions'][0]['ask_know']
                out_nodes = {'yes':'3.3', 'no':'3.4', 'invalid':'-99'}
           
            temp_idx = np.argmax(prediction, axis=-1)
            temp_cls = label2id[temp_idx]
 
            outnode_cls = out_nodes[temp_cls]
            prob = prediction[temp_idx] 
            outype_cls = h_cls[outnode_cls]

            final_result = {'status':0, 'answer':outype_cls, 'score':prob}

    
    return jsonify(final_result)



  
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10025, debug=True)
