from collections import Counter



def convert_by_vocab(vocab, items):
    """
    各种转换, 根据vocab字典类型来决定转换方式
    """
    out = []
    out = [vocab[i] for i in items]
    return out



class FreqVocab:

    def __init__(self,  user_to_list):


        self.counter = Counter()
        self.user_set = set()
        
        for u, item_list in user_to_list.items():
            self.counter.update(item_list) # item_list 每一个元素作为key, value 为计数, 循环更新计数
            self.user_set.add(str(u))
        
        self.user_count = len(self.user_set)
        self.item_count = len(self.counter)
        self.special_tokens = {'[PAD]', '[MASK]', '[NO_USE]'}

        # {'token':'token_的ids, 即频率 频率越高ids越靠前'}
        self.token_to_ids = {} # idx begin from 1

        # 计数器(计数作为 ids)
        for token, count in self.counter.most_common():
            self.token_to_ids[token] = len(self.token_to_ids) + 1
        
        for token in self.special_tokens:
            self.token_to_ids[token] = len(self.token_to_ids) + 1

        for user in self.user_set:
            self.token_to_ids[user] = len(self.token_to_ids) + 1
        
        # {'token_ids':'tokens'}
        self.id_to_tokens = {v:k for k, v in self.token_to_ids.items()}
        # [所有的tokens]
        self.vocab_words = list(self.token_to_ids.keys())


    ## 各种get方法
    def tokens2ids(self, tokens):
        return convert_by_vocab(self.token_to_ids, tokens)
    
    def id2tokens(self, ids):
        return convert_by_vocab(self.id_to_tokens, ids)
    
    def get_vocab_words(self):
        return self.vocab_words
    
    def get_user_count(self):
        return self.user_count
    
    def get_user(self):
        return self.user_set
    
    def get_items_count(self):
        return self.item_count
    
    def get_items(self):
        return list(self.counter.keys())
    
    def get_special_token_count(self):
        return len(self.special_tokens)
    
    def get_special_tokens(self):
        return self.special_tokens

    def get_vocab_size(self):
        return self.get_items_count()+self.get_special_token_count()+1 # start from 1
