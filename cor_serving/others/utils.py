import re


class PreProcessing:


    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    
    def cluster_alphnum(self, text):
        """Simple funtions to aggregate eng and number
    
        Arguments:
            text {str} -- input text
    
        Returns:
            list -- list of string with chinese char or eng word as element
        """
        return_list = []
        last_is_alphnum = False
    
        for char in text:
            is_alphnum = bool(re.match('^[a-zA-Z0-9\[]+$', char))
            is_right_brack = char == ']'
            if is_alphnum:
                if last_is_alphnum:
                    return_list[-1] += char
                else:
                    return_list.append(char)
                    last_is_alphnum = True
            elif is_right_brack:
                if return_list:
                    return_list[-1] += char
                else:
                    return_list.append(char)
                last_is_alphnum = False
            else:
                return_list.append(char)
                last_is_alphnum = False
        return return_list
    
    
    def tokenize_text_with_seqs(self, inputs_a, target, is_seq=False):
        if isinstance(inputs_a, list):
            inputs_a_str = '\t'.join([t if t != '\t' else ' ' for t in inputs_a])
        else:
            inputs_a_str = inputs_a
        if is_seq:
            tokenized_inputs, target = self.tokenizer.tokenize(inputs_a_str, target)
        else:
            tokenized_inputs = self.tokenizer.tokenize(inputs_a_str)
    
        return (tokenized_inputs, target)
    
    
    def truncate_seq_pair(self, tokens_a, tokens_b, target, max_length=36):
        if len(tokens_a) > max_length - 2:
            tokens_a = tokens_a[0:(max_length - 2)]
    
        return tokens_a, tokens_b, target 
    
    
    def add_special_tokens_with_seqs(self, tokens_a, tokens_b, target, is_seq=False):
    
        tokens = []
        segment_ids = []
    
        tokens.append("[CLS]")
        segment_ids.append(0)
    
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
    
        if is_seq:
            target = ['[PAD]'] + target + ['[PAD]']
    
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
    
        return (tokens, segment_ids, target)
    
    
    def create_mask_and_padding(self, tokens, segment_ids, target, max_length=36, is_seq=False, dynamic_padding=False):
    
        input_mask = [1]*len(tokens)
    
        if not dynamic_padding:
            pad_list = ['[PAD]'] * (max_length - len(input_mask))
    
            input_mask += [0]*len(pad_list)
            segment_ids += [0]*len(pad_list)
            tokens += pad_list
    
            if is_seq:
                target += pad_list
    
        return input_mask, tokens, segment_ids, target
    
    
    
    def deal_combine(self, word):
         data_dict = {}
         inputs_a = self.cluster_alphnum(word)
    
         tokens, target = self.tokenize_text_with_seqs( inputs_a, None)
    
         tokens_a, tokens_b, target = self.truncate_seq_pair(tokens, None, target)
    
         tokens, segment_ids, target = self.add_special_tokens_with_seqs(tokens_a, tokens_b, target)
    
         input_mask, tokens, segment_ids, target = self.create_mask_and_padding(tokens, segment_ids, target)
    
         input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
         
         data_dict['input_ids'] = input_ids
         data_dict['input_mask'] = input_mask
         data_dict['segment_ids'] = segment_ids

         return data_dict

