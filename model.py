import torch.nn as nn
import torch

from transformers import AutoConfig, AutoModel


class ExplainableModel(nn.Module):
    def __init__(self, bert_dir):
        super().__init__()
        self.bert_config = AutoConfig.from_pretrained(bert_dir, output_hidden_states=False)
        self.intermediate = AutoModel.from_pretrained(bert_dir)
        self.span_info_collect = SICModel(self.bert_config.hidden_size)
        self.interpretation = InterpretationModel(self.bert_config.hidden_size)
        self.output = nn.Linear(self.bert_config.hidden_size, 3)

    def forward(self, input_ids, start_indexs, end_indexs, span_masks):
        # generate mask
        attention_mask = (input_ids != 1).long()
        # intermediate layer
        hidden_states, first_token = self.intermediate(input_ids, attention_mask=attention_mask)
        # span info collecting layer (SIC)
        h_ij = self.span_info_collect(hidden_states, start_indexs, end_indexs)
        # interpretation layer
        H, a_ij = self.interpretation(h_ij, span_masks)
        #output layer
        out = self.output(H)
        return out, a_ij
        
class SICModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.W_1 = nn.Linear(hidden_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, hidden_size)
        self.W_3 = nn.Linear(hidden_size, hidden_size)
        self.W_4 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states, start_indexs, end_indexs):
        W1_h = self.W_1(hidden_states) # (bs, length, hidden_size)
        W2_h = self.W_2(hidden_states)
        W3_h = self.W_3(hidden_states)
        W4_h = self.W_4(hidden_states)

        W1_hi_emb = torch.index_select(W1_h, 1, start_indexs) # (bs, span_num, hidden_size)
        W2_hj_emb = torch.index_select(W2_h, 1, end_indexs)
        W3_hi_emb = torch.index_select(W3_h, 1, start_indexs)
        W3_hj_emb = torch.index_select(W3_h, 1, end_indexs)
        W4_hi_emb = torch.index_select(W4_h, 1, start_indexs)
        W4_hj_emb = torch.index_select(W4_h, 1, end_indexs)

        # [w1*hi, w2*hj, w3*(hi-hj), w4*(hi⊗hj)]
        span = W1_hi_emb + W2_hj_emb + (W3_hi_emb - W3_hj_emb) + torch.mul(W4_hi_emb, W4_hj_emb)
        h_ij = torch.tanh(span)
        return h_ij

class InterpretationModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.h_t = nn.Linear(hidden_size, 1)

    def forward(self, h_ij, span_masks):
        o_ij = self.h_t(h_ij).squeeze(-1) # (ba, span_num)
        # mask illegal span
        o_ij = o_ij - span_masks
        # normalize all a_ij, a_ij sum = 1
        a_ij = nn.functional.softmax(o_ij, dim=1)
        H = (a_ij.unsqueeze(-1) * h_ij).sum(dim=1) # (bs, hidden_size)
        return H, a_ij