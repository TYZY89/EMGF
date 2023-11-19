import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import os
import pickle
import copy
import numpy as np
from torch_geometric.nn import GCNConv, GCN2Conv, TAGConv, ChebConv, GatedGraphConv, ResGatedGraphConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer


# con, dep and seman update module
class GCNEncoder(nn.Module):
    def __init__(self, emb_dim, con_layers, dep_layers, sem_layers, gcn_dropout=0.1):
        super().__init__()
        self.con_layers = con_layers
        self.dep_layers = dep_layers
        self.sem_layers = sem_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        # gcn layer
        self.W_con = nn.ModuleList()
        self.W_dep = nn.ModuleList()
        self.W_sem = nn.ModuleList()
        for layer in range(self.con_layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W_con.append(nn.Linear(input_dim, input_dim))
        for layer in range(self.dep_layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W_dep.append(nn.Linear(input_dim, input_dim))
        for layer in range(self.sem_layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W_sem.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)

    def forward(self, inputs, con_adj, dep_adj, seman_adj, tok_length):
        # gcn layer
        con_input, dep_input, seman_input = inputs, inputs, inputs

        # denom_con_dep_seman
        dep_denom = dep_adj.sum(2).unsqueeze(2) + 1
        seman_denom = seman_adj.sum(2).unsqueeze(2) + 1

        for index in range(self.con_layers):
            con_adj_new = con_adj[index].bool().float()
            con_denom = con_adj_new.sum(2).unsqueeze(2) + 1

            # con
            con_Ax = con_adj_new.bmm(con_input)
            con_AxW = self.W_con[index](con_Ax)
            con_AxW = con_AxW + self.W_con[index](con_input)  # self loop
            con_AxW = con_AxW / con_denom
            con_gAxW = F.relu(con_AxW)
            con_input = self.gcn_drop(con_gAxW) if index < self.con_layers - 1 else con_gAxW

        for index in range(self.dep_layers):
            # dep
            dep_Ax = dep_adj.bmm(dep_input)
            dep_AxW = self.W_dep[index](dep_Ax)
            dep_AxW = dep_AxW + self.W_dep[index](dep_input)  # self loop
            dep_AxW = dep_AxW / dep_denom
            dep_gAxW = F.relu(dep_AxW)
            dep_input = self.gcn_drop(dep_gAxW) if index < self.dep_layers - 1 else dep_gAxW

        for index in range(self.sem_layers):
            # seman
            seman_Ax = seman_adj.bmm(seman_input)
            seman_AxW = self.W_sem[index](seman_Ax)
            seman_AxW = seman_AxW + self.W_sem[index](seman_input)  # self loop
            seman_AxW = seman_AxW / seman_denom
            seman_gAxW = F.relu(seman_AxW)
            seman_input = self.gcn_drop(seman_gAxW) if index < self.sem_layers - 1 else seman_gAxW

        multi_loss = process_adj_matrices(dep_adj, con_adj[0].bool().float(), seman_adj, con_input, dep_input, tok_length)
        
        return con_input, dep_input, seman_input, multi_loss
    
def process_adj_matrices(dep_adj, con_adj_new, seman_adj, con_input, dep_input, tok_len):
    batch_size, max_length, _ = dep_adj.size()
    
    multi_viewdep_loss = 0

    for b in range(batch_size):
        length = int(tok_len[b])

        # Slice the tensors to the required length
        dep_adj_batch = dep_adj[b, :length, :length].to(dtype=torch.int)
        con_adj_new_batch = con_adj_new[b, :length, :length].to(dtype=torch.int)
        sem_adj_batch = seman_adj[b, :length, :length]

        # caculate importance scores
        node_importance_scores = sem_adj_batch.mean(dim=1) + sem_adj_batch.max(dim=1).values
        k = int(math.log10(length) ** 2)

        top_k_node = torch.topk(node_importance_scores, k).indices.tolist()

        dep_non_zero_indices = torch.nonzero(dep_adj_batch)
        con_non_zero_indices = torch.nonzero(con_adj_new_batch)

        dep_edges_tuple = dep_non_zero_indices.t()
        con_edges_tuple = con_non_zero_indices.t()

        dep_1_start = dep_edges_tuple[0]
        dep_1_end = dep_edges_tuple[1]

        dep_3_start, dep_3_end = get_2nd_order_pairs(con_edges_tuple, dep_edges_tuple)

        multi_Dep_loss = 0

        for i in top_k_node:
            # DEP-CON
            Anchor_Dep = dep_input[b, i]
            AD_dep_view_node_index = dep_1_end[dep_1_start == i]
            AD_con_view_node_index = dep_3_end[dep_3_start == i]

            AD_dep_view_node = dep_input[b, AD_dep_view_node_index]
            AD_remaining_indices_inter = torch.nonzero(~AD_dep_view_node_index.new_full((length,), 0, dtype=torch.bool)).squeeze()
            AD_dep_view_node_remain = dep_input[b, AD_remaining_indices_inter]

            # inter-inter
            AD_con_view_node_index = torch.cat((AD_con_view_node_index, torch.tensor([i]).cuda()))
            AD_con_view_node = con_input[b, AD_con_view_node_index]
            AD_remaining_indices_intra = torch.nonzero(~AD_con_view_node_index.new_full((length,), 0, dtype=torch.bool)).squeeze()
            AD_con_view_node_remain = dep_input[b, AD_remaining_indices_intra]

            # total
            AD_P = torch.cat((AD_dep_view_node, AD_con_view_node), dim=0)
            AD_N = torch.cat((AD_dep_view_node_remain, AD_con_view_node_remain), dim=0)

            AD_loss = multi_margin_contrastive_loss(Anchor_Dep, AD_P, AD_N)

            multi_Dep_loss += AD_loss

        multi_viewdep_loss += multi_Dep_loss

    return multi_viewdep_loss

def multi_margin_contrastive_loss(anchor, positives, negatives, margin=0.2):
    dist_pos = F.pairwise_distance(anchor.unsqueeze(0), positives).mean()
    dist_neg = F.pairwise_distance(anchor.unsqueeze(0), negatives).mean()
    loss = torch.relu(dist_pos - dist_neg + margin) / 10

    return loss

def get_2nd_order_pairs(edge_list1, edge_list2):
    list1, list2 = [], []
    edge_list1 = edge_list1.tolist()  # 转换为Python列表
    edge_list2 = edge_list2.tolist()

    for x0, y0 in zip(edge_list1[0], edge_list1[1]):
        edge_exist = False
        for x1, y1 in zip(edge_list2[0], edge_list2[1]):
            if x1 == x0 and y1 == y0:
                edge_exist = True
                break
        if not edge_exist:
            list1.append(x0)
            list2.append(y0)
    list1, list2 = torch.tensor(list1).cuda(), torch.tensor(list2).cuda()
    list1, list2 = list1.to(torch.int64), list2.to(torch.int64)

    return list1, list2

# condep and seman update module
class ConDep_GCNEncoder(nn.Module):
    def __init__(self, emb_dim=768, num_layers=3,gcn_dropout=0.1):
        super().__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)

    def forward(self, inputs, con_adj, dep_adj, seman_adj):
        # gcn layer
        condep_input, seman_input = inputs, inputs

        # seman denom
        seman_denom = seman_adj.sum(2).unsqueeze(2) + 1

        for index in range(self.layers):
            con_adj_new = con_adj[index].bool().float()
            condep_adj = con_adj_new * dep_adj
            condep_denom = condep_adj.sum(2).unsqueeze(2) + 1

            # condep
            condep_Ax = condep_adj.bmm(condep_input)
            condep_AxW = self.W[index](condep_Ax)
            condep_AxW = condep_AxW + self.W[index](condep_input)  # self loop
            condep_AxW = condep_AxW / condep_denom
            condep_gAxW = F.relu(condep_AxW)
            condep_input = self.gcn_drop(condep_gAxW) if index < self.layers - 1 else condep_gAxW

            # seman
            seman_Ax = seman_adj.bmm(seman_input)
            seman_AxW = self.W[index](seman_Ax)
            seman_AxW = seman_AxW + self.W[index](seman_input)  # self loop
            seman_AxW = seman_AxW / seman_denom
            seman_gAxW = F.relu(seman_AxW)
            seman_input = self.gcn_drop(seman_gAxW) if index < self.layers - 1 else seman_gAxW

        # feature projection loss
        f_p_condep = condep_input
        f_c_seman = seman_input
        f_p_condep_ = proj(f_p_condep, f_c_seman)
        f_pcondep_tilde = proj(f_p_condep, (f_p_condep - f_p_condep_))

        return f_pcondep_tilde, seman_input

# traditional gcn module
class GCN(nn.Module):

    def __init__(self, emb_dim=768, num_layers=2,gcn_dropout=0.1):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)


    def forward(self, inputs, adj):
        # gcn layer
        
        denom = adj.sum(2).unsqueeze(2) + 1
        # mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for l in range(self.layers):
            Ax = adj.bmm(inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](inputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return inputs
    


# multi-head attention layer
class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        # mask = mask[:, :, :query.size(1)]         如果需要mask则使用，否则 # 
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) + 1e-9)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    # p_attn = entmax15(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn



# get embedding
def get_conspan_matrix(span_list, rm_loop=False, max_len=None):
    '''
    span_list: [N,B,L]
    return span:[N,B,L,L]
    '''
    # [N,B,L]
    N, B, L = span_list.shape
    span = get_span_matrix_3D(span_list.contiguous().view(-1, L), rm_loop, max_len).contiguous().view(N, B, L, L)
    return span

def get_span_matrix_3D(span_list, rm_loop=False, max_len=None):
    # [N,L]
    origin_dim = len(span_list.shape)
    if origin_dim == 1:  # [L]
        span_list = span_list.unsqueeze(dim=0)
    N, L = span_list.shape
    if max_len is not None:
        L = min(L, max_len)
        span_list = span_list[:, :L]
    span = span_list.unsqueeze(dim=-1).repeat(1, 1, L)
    span = span * (span.transpose(-1, -2) == span)
    if rm_loop:
        span = span * (~torch.eye(L).bool()).unsqueeze(dim=0).repeat(N, 1, 1)
        span = span.squeeze(dim=0) if origin_dim == 1 else span  # [N,L,L]
    return span

def get_embedding(vocab, opt):
    graph_emb=0

    if opt.dataset == 'laptop':
        graph_file = 'embeddings/entity_embeddings_analogy_400.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_analogy.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_analogy_bert.pkl' % opt.dataset
        # graph_pkl = 'embeddings/%s_graph_analogy_roberta.pkl' % ds_name
    elif opt.dataset == 'restaurant':
        graph_file = 'embeddings/entity_embeddings_distmult_200.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_dismult.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_dismult_bert.pkl' % opt.dataset
        # graph_pkl = 'embeddings/%s_graph_dismult_roberta.pkl' % ds_name
    elif opt.dataset == 'twitter':
        graph_file = 'embeddings/entity_embeddings_distmult_200.txt'
        if opt.is_bert==0:
            graph_pkl = 'embeddings/%s_graph_dismult.pkl' % opt.dataset
        else:
            graph_pkl = 'embeddings/%s_graph_dismult_bert.pkl' % opt.dataset

    if not os.path.exists(graph_pkl):
        graph_embeddings = np.zeros((len(vocab)+1, opt.dim_k), dtype='float32')
        with open(graph_file, encoding='utf-8') as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                graph_emb += 1
                if w in vocab:
                    try:
                        graph_embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        pass
        pickle.dump(graph_embeddings, open(graph_pkl, 'wb'))
    else:
        graph_embeddings = pickle.load(open(graph_pkl, 'rb'))

    return graph_embeddings



def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

def ids2ori_adj(ori_tag, sent_len, head):
    adj = []
    # print(sent_len)
    for b in range(ori_tag.size()[0]):
        ret = np.ones((sent_len, sent_len), dtype='float32')
        fro_list = head[b]
        for i in range(len(fro_list) - 1):
            to = i + 1
            fro = fro_list[i]
            ret[fro][to] = ori_tag[b][i]
            ret[to][fro] =ori_tag[b][i]
        adj.append(ret)

    return adj

def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i in range(batch_size):
        sequence_mask[i, :lengths[i]] = True

    return sequence_mask.cuda()

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        self.RNN.flatten_parameters()
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        if self.rnn_type == 'LSTM': 
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else: 
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)
        
class ConvInteract(nn.Module):
    def __init__(self, hidden_dim):
        super(ConvInteract, self).__init__()

        self.hidden_dim = hidden_dim

        # Define residual connections and LayerNorm layers
        self.residual_layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        
        # self.linear = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.GatedGCN = GatedGCN(hidden_dim, hidden_dim)

        # Fusion layer
        self.lstm = nn.LSTM(self.hidden_dim*2, self.hidden_dim, 2, batch_first=True,
                            bidirectional=True)

        # MLP
        self.feature_fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))

    def forward(self, h_feature, h_con, h_dep, h_seman):

        h_con_inter, h_dep_inter, h_seman_inter = self.GatedGCN(h_con, h_dep, h_seman)

        # residual enhanced layer
        h_con_new = self.residual_layer1(h_feature + h_con_inter)
        h_dep_new = self.residual_layer2(h_feature + h_dep_inter)
        h_seman_new = self.residual_layer3(h_feature + h_seman_inter)

        # concat = torch.cat([h_syn_feature, h_sem_feature], dim=2)
        # output, _ = self.lstm(concat)
        # h_fusion = self.feature_fusion(output)

        return h_con_new, h_dep_new, h_seman_new
    
class FeatureStacking(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureStacking, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input1, input2):
        # stack the three input features along the third dimension to form a new tensor with dimensions [a, b, c, hidden_dim]
        # stacked_input = torch.stack([input1, input2, input3, input4], dim=3)
        stacked_input = torch.stack([input1, input2], dim=3)

        # apply average pooling along the fourth dimension to obtain a tensor with dimensions [a, b, c, 1]
        # pooled_input = torch.mean(stacked_input, dim=3, keepdim=True)
        pooled_input,_ = torch.max(stacked_input, dim=3, keepdim=True)

        # reshape the tensor to the desired output shape [a, b, c]
        output = pooled_input.squeeze(3)

        return output

class GatedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, gated_layers=2):
        super(GatedGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gated_layers = gated_layers
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)                                                           # GCNConv默认添加add_self_loops
        # self.conv2 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv3 = GatedGraphConv(self.hidden_dim, self.gated_layers)

    def forward(self, h_con, h_dep, h_seman):
        # Build graph data structures
        B, S, D = h_con.shape
        h_con_ = h_con.view(-1)
        h_dep_ = h_dep.view(-1)
        h_seman_ = h_seman.view(-1)
        features = torch.stack([h_con_, h_dep_, h_seman_], dim=-1)
        data = Data(x=features)
        data.cuda()
        data.x = data.x.view(-1, self.input_dim)
        data.edge_index, _ = dense_to_sparse(torch.ones(S, S).cuda())
        data.edge_attr = compute_cosine_similarity(data.x, data.edge_index)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index))

        h_fusion_con, h_fusion_dep, h_fusion_seman = x.view(3, B, S, D)[0], x.view(3, B, S, D)[1], x.view(3, B, S, D)[2]

        return h_fusion_con, h_fusion_dep, h_fusion_seman
    
# Calculate the edge weights, i.e. Euclidean distance
def edge_weight(x, edge_index):
    row, col = edge_index
    edge_attr = (x[row] - x[col]).norm(p=2, dim=-1).view(edge_index.size(1), -1)

    return edge_attr

# Cosine similarity
def compute_cosine_similarity(x, edge_index):

    edge_index_row, edge_index_col = edge_index[0], edge_index[1]

    x_row = x[edge_index_row]
    x_col = x[edge_index_col]
    similarity = F.cosine_similarity(x_row, x_col, dim=1)
    min_value = similarity.min()
    max_value = similarity.max()
    similarity = (similarity - min_value) / (max_value - min_value)

    return similarity

# Pearson correlation coefficient
def compute_pearson_correlation(x, edge_index):
    mean_x = torch.mean(x, dim=1)

    # Compute differences between each value and the mean for x
    diff_x = x - mean_x[:, None]

    # Compute the sum of squared differences for x
    sum_squared_diff_x = torch.sum(diff_x ** 2, dim=1)

    # Compute the square root of the sum of squared differences for x
    sqrt_sum_squared_diff_x = torch.sqrt(sum_squared_diff_x)

    # Compute the product of the square roots for x
    product_sqrt_diff_x = sqrt_sum_squared_diff_x[edge_index[0]] * sqrt_sum_squared_diff_x[edge_index[1]]

    # Compute the sum of the multiplied differences
    sum_multiplied_diff = torch.sum(diff_x[edge_index[0]] * diff_x[edge_index[1]], dim=1)

    # Compute the Pearson correlation coefficient
    pearson_corr = sum_multiplied_diff / product_sqrt_diff_x

    return pearson_corr

# Interact Module(EMFH)
class ResEMFH(nn.Module):
    def __init__(self, opt, d_bert):
        super(ResEMFH, self).__init__()
        self.opt = opt

        if self.opt.high_order:
            self.mfh1 = MFB(opt, False, d_bert)
            self.mfh2 = MFB(opt, False, d_bert)
            self.mfh3 = MFB(opt, False, d_bert)
            self.mfh4 = MFB(opt, False, d_bert)
            self.mfh5 = MFB(opt, False, d_bert)
            self.mfh6 = MFB(opt, False, d_bert)
        else:
            self.mfb = MFB(opt, True, d_bert)

    def forward(self, con_feat, dep_feat, sem_feat, know_feat):

        if self.opt.high_order:
            z1, exp1, f_pcon_1, f_pdep_1 = self.mfh1(con_feat, dep_feat, sem_feat, know_feat)
            residual1 = z1
            z2, exp2, f_pcon_2, f_pdep_2 = self.mfh2(f_pcon_1, f_pdep_1, sem_feat, know_feat, exp1)
            residual2 = z2
            z3, exp3, f_pcon_3, f_pdep_3 = self.mfh3(f_pcon_2, f_pdep_2, sem_feat, know_feat, exp2)
            residual3 = z3
            z4, exp4, f_pcon_4, f_pdep_4 = self.mfh4(f_pcon_3, f_pdep_3, sem_feat, know_feat, exp3)
            residual4 = z4
            z5, exp5, f_pcon_5, f_pdep_5 = self.mfh5(f_pcon_4, f_pdep_4, sem_feat, know_feat, exp4)
            residual5 = z5
            z6, exp6, f_pcon_6, f_pdep_6 = self.mfh6(f_pcon_5, f_pdep_5, sem_feat, know_feat, exp5)
            z6 = z6 + residual3
            z = torch.mean(torch.cat((z1, z2, z3, z4, z5, z6), 1), dim=1, keepdim=False)
        else:
            z, _ = self.mfb(con_feat, dep_feat, sem_feat, know_feat)
            z = z.squeeze(1)

        return z
    
class MFB(nn.Module):
    def __init__(self, opt, is_first, d_bert):
        super(MFB, self).__init__()
        self.opt = opt
        self.is_first = is_first
        self.bert_dim = d_bert

        # Proj layer
        self.proj_i = nn.Linear(d_bert, d_bert)
        self.proj_q = nn.Linear(d_bert, d_bert)
        self.proj_ent = nn.Linear(d_bert, d_bert)
        self.proj_struct = nn.Linear(2*opt.lstm_dim+opt.dim_k, d_bert)

        # dropout layer
        self.dropout = nn.Dropout(opt.dropout_r)

    def forward(self, con_feat, dep_feat, sem_feat, know_feat, exp_in=1):
        # Expand Stage
        batch_size = con_feat.shape[0]
        con_feat = self.proj_i(con_feat)
        dep_feat = self.proj_q(dep_feat)
        sem_feat = self.proj_ent(sem_feat)
        know_feat = self.proj_struct(know_feat)

        exp_out = con_feat * dep_feat * sem_feat * know_feat
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)

        # Suqeeze Stage
        z = torch.sqrt(F.relu(exp_out)) - torch.sqrt(F.relu(-exp_out))
        z = F.normalize(z.view(batch_size, -1))
        z = z.view(batch_size, -1, self.bert_dim)

        # orthogonal projection
        f_p_con = con_feat
        f_c_seman = sem_feat
        f_p_con_ = proj(f_p_con, f_c_seman)
        f_pcon_tilde = proj(f_p_con, (f_p_con - f_p_con_))

        f_p_dep = dep_feat
        f_c_seman = sem_feat
        f_p_dep_ = proj(f_p_dep, f_c_seman)
        f_pdep_tilde = proj(f_p_dep, (f_p_dep - f_p_dep_))
        
        return z, exp_out, f_pcon_tilde, f_pdep_tilde
    
def proj(x, y):
    numerator = x * y
    denominator = torch.abs(y) + torch.finfo(torch.float32).eps  
    projection = numerator / denominator * (y / denominator)
    
    return projection

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))