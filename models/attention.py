import math
import torch
import torch.nn as nn
from numpy.core.arrayprint import set_printoptions
from torch.nn import CrossEntropyLoss, MSELoss,BCELoss
import numpy as np
np.set_printoptions(threshold=np.inf)
# import pandas
# pandas.set_option('display.max_rows',None)

def _getMatrixTree_multi(scores, root):
    A = scores.exp()  #A_ij
    R = root.exp()    #exp(r_i)

    L = torch.sum(A, 1) #A_i'j 所有以j为根的依赖边之和
    L = torch.diag_embed(L) #创建L的对角矩阵
    L = L - A
    LL = L + torch.diag_embed(R) #当i=j的对角线上加上exp(ri)


    LL_inv = torch.inverse(LL)  # batch_l, doc_l, doc_l   # L^-1
    # print(LL_inv.shape)
    LL_inv_diag = torch.diagonal(LL_inv, 0, 1, 2) #求每个batch的对角线
    d0 = R * LL_inv_diag  #P_{i}^r
    LL_inv_diag = torch.unsqueeze(LL_inv_diag, 2)

    _A = torch.transpose(A, 1, 2) #A转置
    _A = _A * LL_inv_diag
    tmp1 = torch.transpose(_A, 1, 2)
    tmp2 = A * torch.transpose(LL_inv, 1, 2)

    d = tmp1 - tmp2  #P_ij
    # print(d.shape)

    # sentence1 = d[0].tolist()
    # f = open('test.txt', 'w+')
    # for i in range(22):
    #     for j in range(22):
    #         num = sentence1[i][j]
    #         f.write(str(num))
    #         f.write(' ')
    #     f.write('\n')
    # f.close()

    #
    # print("Pij = ",d[1])
    # # print("Pij = ",d[1])
    # # print("Pij = ",d[2])
    # breakpoint()

    return d, d0

class StructuredAttention(nn.Module):
    def __init__(self, opt):
        self.model_dim = opt.bert_dim

        super(StructuredAttention, self).__init__()

        # self.linear_keys = nn.Linear(opt.bert_dim, self.model_dim)
        # self.linear_query = nn.Linear(opt.bert_dim, self.model_dim)
        self.linear_root = nn.Linear(opt.bert_dim, 1) #

        self.dropout = nn.Dropout(opt.bert_dropout)


    def forward(self, x, adj, mask = None,roots_label=None,root_mask=None):

        # structured_output, adj_latent, loss_root = self.str_att(
        #     gcn_inputs, dep_adj, extended_attention_mask, aspect_mask, root_mask)

        # key = self.linear_keys(x)
        # query = self.linear_query(x)
        root= self.linear_root(x).squeeze(-1)   #r_i
        # query = query / math.sqrt(self.model_dim)
        # scores = torch.matmul(query, key.transpose(1, 2))  #e_ij
        
        #使用概率矩阵做e_ij
        scores = adj #[B,L,L]


        mask=mask.squeeze(1)/-10000
        root = root - mask.squeeze(1) * 50
        root = torch.clamp(root, min=-40)
        scores = scores - mask * 50
        scores = scores - torch.transpose(mask, 1, 2) * 50
        scores = torch.clamp(scores, min=-40)

        d, d0 = _getMatrixTree_multi(scores, root) # (P_i^r)d0-> B,L   (P_ij)d->B,L,L

        if roots_label is not None:

            loss_fct=BCELoss(reduction='none')
            if root_mask is not None:
                active_loss = root_mask.contiguous().view(-1) == 1
                active_logits = d0.view(-1)

                active_labels = torch.where(
                    active_loss, roots_label.contiguous().view(-1), torch.tensor(0.).type_as(roots_label)
                )

                active_logits=torch.clamp(active_logits,1e-5,1 - 1e-5)

                active_logits = active_logits.to(torch.float32)
                active_labels = active_labels.to(torch.float32)

                loss_root = loss_fct(active_logits, active_labels)

                loss_root = (loss_root*root_mask.contiguous().view(-1).float()).mean()

        attn = torch.transpose(d, 1,2)#
        if mask is not None:
            mask = mask.expand_as(scores).bool()
            attn = attn.masked_fill(mask, 0)

        # print(attn.shape)  #batch, seq_len, seq
        # breakpoint()
        # attn = attn.to(torch.double)
        # x = x.to(torch.double)
        # print(attn)
        context = torch.matmul(attn,x) #[:0]
        # print("context : ", context)
        # print(context.shape)  #(batch, seq, hidden)
        # print(x.shape)  #batch, seq_len, hidden
        # print(attn[1].tolist())
        # sentence1 = attn[1].tolist()
        # f = open('test.txt', 'w+')
        # for i in range(96):
        #     for j in range(96):
        #         num = sentence1[i][j]
        #         f.write(str(num))
        #         f.write(' ')
        #     f.write('\n')
        # f.close()
        # breakpoint()

        return context, d,loss_root
