# -*- coding: utf-8 -*-

cover_pair = [
    ["hawe", "huawei"],
    ["hawei", "huawei"],
    ["hawuia", "huawei"],
    ["lphone", "iphone"],
    ["bluethoot", "bluetooth"],
    ["redime", "redmi"],
    ["remidi", "redmi"],
    ["remi", "redmi"],
    ["samson", "samsung"],
    ["sunsang", "samsung"],
    ["iphones", "iphone"],
    ["iphonex", "iphone x"],
    ["iphonexs", "iphone xs"],
    ["samusge", "samsung"],
    ["samusug", "samsung"],
    ["samsug", "samsung"],
    ["xioami", "xiaomi"],
    ["rechageable", "rechargeable"],
    ["rrchargeable", "rechargeable"],
    ["bankpower", "powerbank"],
]

cover_wrong_words = [x[0] for x in cover_pair]
cover_dict = dict(cover_pair)

replace_arr = [
    ["i phone", "iphone"],
    ["iphonex", "iphone x"],
    ["bank power", "powerbank"],
]

meaningless_words = ['about', 'above', 'across', 'add', 'after', 'against', 'all', 'also', 'always',
                     'among', 'an', 'and', 'any', 'anywhere', 'are', 'as', 'at', 'back', 'be', 'because',
                     'before', 'behind', 'beneath', 'beside', 'between', 'beyond', 'body', 'but', 'by',
                     'call', 'can', 'comes', 'concerning', 'despite', 'does', 'down', 'during', 'end',
                     'even', 'except', 'find', 'fit', 'for', 'fr', 'from', 'full', 'function', 'furthermore',
                     'get', 'give', 'has', 'have', 'i', 'idx', 'if', 'in', 'inside', 'into', 'is', 'it', 'item',
                     'its', 'just', 'keep', 'led', 'look', 'make', 'many', 'me', 'mm', 'my', 'myself', 'nbsp',
                     'need', 'never', 'no', 'non', 'nor', 'not', 'now', 'of', 'on', 'once', 'only', 'onto', 'or',
                     'other', 'our', 'out', 'outside', 'own', 'past', 'play', 'range', 'same', 'set', 'should',
                     'so', 'some', 'take', 'tf', 'than', 'thank', 'that', 'the', 'then', 'there', 'these',
                     'they', 'this', 'through', 'to', 'top', 'true', 'under', 'until', 'up', 'very',
                     'was', 'we', 'were',  'when', 'which', 'with', 'within', 'without', 'you', 'your']
