# -*- coding: utf-8 -*-
import elasticsearch
root_index = "b2c_algo_es"
ELASTICSEARCH_HOST = ["http://127.0.0.1:9200"]

if __name__ == "__main__":
    # os.system("""
    # curl -X GET "localhost:9200/b2c_algo_es/_search?pretty" -H 'Content-Type: application/json' -d'
    # {
    #     "query": {
    #         "match": {
    #             "good_name": {
    #                 "query":                "xiaomi",
    #                 "minimum_should_match": "50%"
    #             }
    #         }
    #     },
    #     "size": 5
    # }
    # '
    # """)
    es = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)
    res = es.search(index=root_index, size=20, body={
        "query": {
            "match": {
                "good_name": {
                    "query":                "xiaomi",
                    "minimum_should_match": "50%"
                }
            }
        },
        "size": 5
    })
    print res
