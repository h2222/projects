#import os, sys
#sys.path.append('/home/jiaxiang.hao/NLP/multi_bert_record_win')
import aiohttp, asyncio
import time

from others import (PreProcessing, FullTokenizer)




async def get_html(url):
    print('start get url')
    # 为了模拟模型请求时间, 我们认为让模型请求任务暂停两秒2秒
    await asyncio.sleep(2)
    print('end get url')




def test():
    # 开始时间
    start_time = time.time()
    # 启动时间循环(一个线程)
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    # 建立事件循环

    # 多任务并发, 不管你设置range为 10 还是 100, 最后结果执行2秒
    tasks = [get_html('baidu.com') for i in range(10)]
    # 执行事件循环并返回结果
    event_loop.run_until_complete(asyncio.wait(tasks))
    
    # 打印结束时间
    print('时间为:', time.time() - start_time)



  
if __name__ == "__main__":
    pass
    # 结果都为两秒
    #test()
