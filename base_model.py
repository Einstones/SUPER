import torch
import torch.nn as nn
import os
import numpy as np
import random
from modules.language_model import WordEmbedding, QuestionEmbedding
from .EuLER import CRLayer

def set_seed():
        seed = 22
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

class EuLER(nn.Module):
    def __init__(self, dataset, k_att, w_emb, q_emb, k_emb, layers=10, num_hid=512, mid_size=512, dropout=0.1, flat_glimpse=1):
        super(EuLER, self).__init__()
        self.img_feat_linear = nn.Linear(4 * num_hid, num_hid)
        self.moe_block = CRLayer(num_hid, 36, layers)
        self.layers = layers
        self.k_att = k_att
        self.k_emb = k_emb
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.num_cell = 5
        self.classifier = nn.Linear(2 * mid_size, dataset.num_ans_candidates)

    def forward(self, v, b, k, q, labels):
        """
           v: (bs, num_r, 512)
           q: (bs, q_len)
           k: (bs, k_len, 1024)
        """
        # Visual and Textual Encoder
        v_emb = self.img_feat_linear(v)  # (bs, num_r, 512)
        w_emb = self.w_emb(q)
        k_emb = self.k_emb(k)  # [bs, k_len, 512]
        q_emb = self.q_emb.forward_all(w_emb)  # [bs, q_len, 512]
        last_in = [v_emb] * self.num_cell

        for k_enc in self.k_att:
            k_emb = k_enc(k_emb)

        # EuLER
        att = []
        mm = torch.zeros_like(v_emb)
        for i in range(self.layers):
            (last_in, mm, att_map) = self.moe_block(last_in, mm, k_emb, q_emb, i)
            if self.training == False:
                att.append(att_map.detach().cpu())

        # Classifier
        q_feat = torch.mean(q_emb)
        v_feat = torch.mean(mm)
        logits = self.classifier(q_feat + v_feat)

        if self.training == False:
            for i, item in enumerate(att):
                if i == 0:
                    out = item
                else:
                    out = torch.cat([out, item], dim=0)
            return logits, out # (5, 25)
        else:
            return logits


def build_bgns(dataset, num_hid, op='', T_ctrl=8):
    k_att = nn.ModuleList([SA(num_hid) for _ in range(1)])
    k_emb = nn.Linear(2 * num_hid, num_hid)  # [32, 1024]
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, op)
    q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    return EuLER(dataset, k_att, w_emb, q_emb, k_emb, T_ctrl, num_hid)



