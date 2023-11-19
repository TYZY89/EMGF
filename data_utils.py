import os
import sys

# from pygments import lex
sys.path.append(r'../LAL-Parser/src_joint')
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch

class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt, dep_vocab):
        # load raw data
        with open(fname, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        self.data = self.process(raw_data, tokenizer, opt, dep_vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def process(self, raw_data, tokenizer, opt, dep_vocab):

        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        max_len = opt.max_length 
        CLS_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
        SEP_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
        sub_len = len(opt.special_token)
        processed = []

        for d in raw_data:
            tok = d['token']
            if opt.lower:
                tok = [t.lower() for t in tok]
            text_raw_bert_indices, word_mapback, _ = text2bert_id(tok, tokenizer)
            text_raw_bert_indices = text_raw_bert_indices[:max_len]
            word_mapback = word_mapback[:max_len]

            # 1、tok_length
            tok_length = word_mapback[-1] + 1

            # 2、bert_length
            bert_length = len(word_mapback) 

            # 3、dep_spans
            dep_head = list(d['dep_head'])[:tok_length]
            dep_spans = head_to_adj_oneshot(dep_head, tok_length, d['aspects'])

            # con_spans
            con_head = d['con_head']
            con_mapnode = d['con_mapnode']
            con_path_dict, con_children = get_path_and_children_dict(con_head)
            mapback = [ idx for idx ,word in enumerate(con_mapnode) if word[-sub_len: ]!= opt.special_token]
            layers, influence_range, node2layerid = form_layers_and_influence_range(con_path_dict, mapback)
            spans = form_spans(layers, influence_range, tok_length, con_mapnode)

            # parameters initial
            bert_sequence_list = []
            bert_segments_ids_list = []
            polarity_list = []
            aspect_mask_list = []
            aspect_token_list = []
            src_mask_list = []
            con_spans_list = []

            for aspect in d['aspects']:
                asp = list(aspect['term'])
                asp_bert_ids, _, _ = text2bert_id(asp, tokenizer)
                bert_sequence = CLS_id  + text_raw_bert_indices +  SEP_id + asp_bert_ids + SEP_id
                bert_segments_ids = [0] * (bert_length + 2) + [1] * (len(asp_bert_ids ) +1)

                # 4、bert_sequence
                bert_sequence = bert_sequence[:max_len+3]

                # 5、bert_segments_ids
                bert_segments_ids = bert_segments_ids[:max_len+3]

                # 6、polarity
                polarity = polarity_dict[aspect['polarity']]

                # 7、aspect_mask
                term_start = aspect['from']
                term_end = aspect['to']
                aspect_mask = [0] * tok_length
                for pidx in range(term_start, term_end):
                    aspect_mask[pidx] = 1

                # 8、con_spans
                aspect_range = list(range(mapback[aspect['from']], mapback[aspect['to']-1] + 1))
                con_lca = find_inner_LCA(con_path_dict, aspect_range)
                select_spans, span_indications = form_aspect_related_spans(con_lca, spans, con_mapnode, node2layerid, con_path_dict)
                select_spans = select_func(select_spans, opt.max_num_spans, tok_length)
                con_spans = [[ x+ 1 for x in span] for span in select_spans] 

                # 9、src_mask
                src_mask = [1] * tok_length

                # combine
                bert_sequence_list.append(bert_sequence)
                bert_segments_ids_list.append(bert_segments_ids)
                polarity_list.append(polarity)
                aspect_mask_list.append(aspect_mask)
                aspect_token_list.append(asp_bert_ids)
                src_mask_list.append(src_mask)
                con_spans_list.append(con_spans) 
            
            processed += [
                (
                    tok_length, bert_length, bert_sequence_list, bert_segments_ids_list, polarity_list,
                    aspect_token_list, aspect_mask_list, src_mask_list, con_spans_list, dep_spans, word_mapback
                )
            ]
        return processed

def ABSA_collate_fn(batch):
    batch_size = len(batch)
    batch = list(zip(*batch))

    (tok_length_, bert_length_, bert_sequence_list_, bert_segments_ids_, polarity_list_,
                    aspect_token_list_, aspect_mask_list_, src_mask_list_, con_spans_list_, dep_spans_, word_mapback_) = batch
    
    # sequence max length
    lens = batch[0]
    max_lens = max(lens)
    
    # tok_length
    tok_length = torch.LongTensor(tok_length_)

    # bert_length
    bert_length = torch.LongTensor(bert_length_)

    # word_mapback
    word_mapback = get_long_tensor(word_mapback_, batch_size)

    # dep_spans
    dep_spans = np.zeros((batch_size, max_lens, max_lens), dtype=np.float32)
    for idx in range(batch_size):
        mlen = dep_spans_[idx].shape[0]
        dep_spans[idx,:mlen,:mlen] = dep_spans_[idx]
    dep_spans = torch.FloatTensor(dep_spans)

    # as_batch_size
    map_AS = [[idx] * len(a_i) for idx, a_i in enumerate(bert_sequence_list_)]
    map_AS_idx = [range(len(a_i)) for a_i in bert_sequence_list_]
    map_AS = torch.LongTensor([m for m_list in map_AS for m in m_list])
    map_AS_idx = torch.LongTensor([m for m_list in map_AS_idx for m in m_list])
    as_batch_size = len(map_AS)

    # bert_sequence_list
    bert_sequence = [p for p_list in bert_sequence_list_ for p in p_list]
    bert_sequence = get_long_tensor(bert_sequence, as_batch_size)

    # bert_segments_ids
    bert_segments_ids = [p for p_list in bert_segments_ids_ for p in p_list]
    bert_segments_ids = get_long_tensor(bert_segments_ids, as_batch_size)

    # polarity
    polarity = torch.LongTensor([sl for sl_list in polarity_list_ for sl in sl_list if isinstance(sl, int)])

    # aspect_token_list
    aspect_token_list = [p for p_list in aspect_token_list_ for p in p_list]
    aspect_token_list = get_long_tensor(aspect_token_list, as_batch_size)

    # aspect_mask
    aspect_mask_list = [p for p_list in aspect_mask_list_ for p in p_list]
    aspect_mask = get_long_tensor(aspect_mask_list, as_batch_size)

    # src_mask
    src_mask_list = [p for p_list in src_mask_list_ for p in p_list]
    src_mask = get_long_tensor(src_mask_list, as_batch_size)

    # con_spans
    con_spans_list = [p for p_list in con_spans_list_ for p in p_list]
    max_num_spans = max([len(p) for p in con_spans_list])
    con_spans = np.zeros((as_batch_size, max_num_spans, max_lens), dtype=np.int64)
    for idx in range(as_batch_size):
        mlen = len(con_spans_list[idx][0])
        con_spans[idx,:,:mlen] = con_spans_list[idx]
    con_spans = torch.LongTensor(con_spans)

    return  (
        tok_length, bert_length, bert_sequence, bert_segments_ids, word_mapback, map_AS,\
        aspect_token_list, aspect_mask, src_mask, dep_spans, con_spans, polarity
    )

    
def text2bert_id(token, tokenizer):
    re_token = []
    word_mapback = []
    word_split_len = []
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))
        word_split_len.append(len(temp))
    re_id = tokenizer.convert_tokens_to_ids(re_token)
    return re_id ,word_mapback, word_split_len

def get_path_and_children_dict(heads):
    path_dict = {}
    remain_nodes = list(range(len(heads)))
    delete_nodes = []
    
    while len(remain_nodes) > 0:
        for idx in remain_nodes:
            #初始状态
            if idx not in path_dict:
                path_dict[idx] = [heads[idx]]  # no self
                if heads[idx] == -1:
                    delete_nodes.append(idx) #need delete root
            else:
                last_node = path_dict[idx][-1]
                if last_node not in remain_nodes:
                    path_dict[idx].extend(path_dict[last_node])
                    delete_nodes.append(idx)
                else:
                    path_dict[idx].append(heads[last_node])
        #remove nodes
        for del_node in delete_nodes:
            remain_nodes.remove(del_node)
        delete_nodes = []

    #children_dict
    children_dict = {}
    for x,l in path_dict.items():
        if l[0] == -1:
            continue
        if l[0] not in children_dict:
            children_dict[l[0]] = [x]
        else:
            children_dict[l[0]].append(x)

    return path_dict, children_dict

def form_spans(layers, influence_range, token_len, con_mapnode, special_token = '[N]'):
    spans = []
    sub_len = len(special_token)
    
    for _, nodes in layers:

        pointer = 0
        add_pre = 0
        temp = [0] * token_len
        temp_indi = ['-'] * token_len
        
        for node_idx in nodes:
            begin,end = influence_range[node_idx] 
            
            if con_mapnode[node_idx][-sub_len:] == special_token:
                temp_indi[begin:end] = [con_mapnode[node_idx][:-sub_len]] * (end-begin)
            
            if(begin != pointer): 
                sub_pre = spans[-1][pointer] 
                temp[pointer:begin] = [x + add_pre-sub_pre for x in spans[-1][pointer:begin]] #
                add_pre = temp[begin-1] + 1
            temp[begin:end] = [add_pre] * (end-begin)  

            add_pre += 1
            pointer = end
        if pointer != token_len: 
            sub_pre = spans[-1][pointer]
            temp[pointer:token_len] = [x + add_pre-sub_pre for x in spans[-1][pointer:token_len]]
            add_pre = temp[begin-1] + 1
        spans.append(temp)

    return spans

def form_layers_and_influence_range(path_dict,mapback):
    sorted_path_dict = sorted(path_dict.items(),key=lambda x: len(x[1]))
    influence_range = { cid:[idx,idx+1] for idx,cid in enumerate(mapback) }
    layers = {}
    node2layerid = {}
    for cid,path_dict in sorted_path_dict[::-1]:
    
        length = len(path_dict)-1
        if length not in layers:
            layers[length] = [cid]
            node2layerid[cid] = length
        else:
            layers[length].append(cid)
            node2layerid[cid] = length
        father_idx = path_dict[0]
        
        
        assert(father_idx not in mapback)
        if father_idx not in influence_range:
            influence_range[father_idx] = influence_range[cid][:] #deep copy
        else:
            influence_range[father_idx][0] = min(influence_range[father_idx][0], influence_range[cid][0])
            influence_range[father_idx][1] = max(influence_range[father_idx][1], influence_range[cid][1])  
    
    layers = sorted(layers.items(),key=lambda x:x[0])
    layers = [(cid,sorted(l)) for cid,l in layers]  # or [(cid,l.sort()) for cid,l in layers]

    return layers, influence_range,node2layerid

def select_func(spans, max_num_spans, length):
    if len(spans) <= max_num_spans:
        lacd_span = spans[-1] if len(spans) > 0 else [0] * length
        select_spans = spans + [lacd_span] * (max_num_spans - len(spans))

    else:
        if max_num_spans == 1:
            select_spans = spans[0] if len(spans) > 0 else [0] * length
        else:
            gap = len(spans)  // (max_num_spans-1)
            select_spans = [ spans[gap * i] for i in range(max_num_spans-1)] + [spans[-1]]

    return select_spans

def head_to_adj_oneshot(heads, sent_len, aspect_dict,
                        leaf2root=True, root2leaf=True, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)

    heads = heads[:sent_len]

    # aspect <self-loop>
    for asp in aspect_dict:
        from_ = asp['from']
        to_ = asp['to']
        for i_idx in range(from_, to_):
            for j_idx in range(from_, to_):
                adj_matrix[i_idx][j_idx] = 1

    for idx, head in enumerate(heads):
        if head != -1:
            if leaf2root:
                adj_matrix[head, idx] = 1
            if root2leaf:
                adj_matrix[idx, head] = 1

        if self_loop:
            adj_matrix[idx, idx] = 1

    return adj_matrix

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens

def find_inner_LCA(path_dict,aspect_range):
    path_range = [ [x] + path_dict[x] for x in aspect_range]
    path_range.sort(key=lambda l:len(l))
 
    for idx in range(len(path_range[0])):
        flag = True
        for pid in range(1,len(path_range)):
            if path_range[0][idx]  not in path_range[pid]:
                flag = False #其中一个不在
                break
            
        if flag: #都在
            LCA_node = path_range[0][idx]
            break #already find
    return LCA_node

def form_aspect_related_spans(aspect_node_idx, spans, mapnode, node2layerid, path_dict,select_N = ['ROOT','TOP','S','NP','VP'], special_token = '[N]'):
    aspect2root_path = path_dict[aspect_node_idx]
    span_indications = []
    spans_range = []
    
    for idx,f in enumerate(aspect2root_path[:-1]):
        if mapnode[f][:-len(special_token)] in select_N:
            span_idx = node2layerid[f]
            span_temp = spans[span_idx]

            if len(spans_range) == 0 or span_temp != spans_range[-1]:
                spans_range.append(span_temp)
                span_indications.append(mapnode[f][:-len(special_token)])
        
    return spans_range, span_indications

    
def build_senticNet():
    file_path = ['./dataset/opinion_lexicon/SenticNet/negative.txt',
                 './dataset/opinion_lexicon/SenticNet/positive.txt']
    datalist1 = [x.strip().split('\t') for x in open(file_path[0]).readlines()]
    datalist2 = [x.strip().split('\t') for x in open(file_path[1]).readlines()]
    data_list = datalist1 + datalist2
    lexicon_dict = {}
    for key, val in data_list:
        lexicon_dict[key] = abs(float(val))
        # lexicon_dict[key] = float(val)
    return lexicon_dict
    