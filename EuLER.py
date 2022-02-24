import torch.nn as nn
import torch, math

class Path_GRU(nn.Module):
    def __init__(self, num_hid=512):
        super(Path_GRU, self).__init__()
        self.fc1 = nn.Linear(num_hid, num_hid)
        self.fc2 = nn.Linear(num_hid, num_hid)
        self.gate = nn.Sequential(nn.Linear(num_hid, num_hid), nn.ReLU(), nn.Linear(num_hid, 2))

    def forward(self, new, old):
        gate = torch.sigmoid(self.gate(new - old))  # [bs, num_r, num_hid]
        reset_gate, update_gate = gate.chunk(2, 2)  # 2 * (bs, num_r, num_hid)
        res = self.fc1(reset_gate * old) + self.fc2(new)
        out = (1 - update_gate) * old + update_gate * res
        return out

class CRLayer(nn.Module):
    def __init__(self, num_hid=512, num_r=36, steps=10):
        super(CRLayer, self).__init__()
        self.cells = nn.ModuleList([AKECell(num_hid), ICell(num_hid), LRCell(num_hid), SACell(num_hid), GRCell(num_hid)])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.num_hid = num_hid
        self.glimpse = 1
        self.num_cell = 5
        self.num_r = num_r
        self.gru = Path_GRU(num_hid)
        self.fc_ake = nn.Linear(num_hid, self.num_cell * self.glimpse)
        self.fc_i = nn.Linear(num_hid, self.num_cell * self.glimpse)
        self.fc_sa = nn.Linear(num_hid, self.num_cell * self.glimpse)
        self.fc_lr = nn.Linear(num_hid, self.num_cell * self.glimpse)
        self.fc_gr = nn.Linear(num_hid, self.num_cell * self.glimpse)
        self.gate = nn.Sequential(nn.Linear(num_hid, num_hid), nn.ReLU(), nn.Linear(num_hid, 2 * num_hid))

    def squash(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        scale = (norm ** 2) / (1 + norm ** 2)
        return scale * x

    def forward(self, last_in, last_mm, k_emb, q, i):

        # Modular Network
        res = []
        for k, modular in enumerate(self.cells):
            res.append(modular(last_in[k], q, k_emb))

        # Memory, Squash function
        mm = torch.zeros_like(res[0])
        for j in range(self.num_cell):
            mm += self.squash(res[j] * last_mm)  # 进行压缩
        mm = mm / self.num_cell

        # Routing Network
        path_ake = self.fc_ake((res[0] + last_mm).mean(dim=1))
        path_ake_sm = torch.sigmoid(path_ake) # [bs, 8, 4]
        path_ake_att = path_ake_sm.unsqueeze(dim=1).repeat(1, self.num_r, 1)

        path_i = self.fc_i((res[1] + mm).mean(dim=1))
        path_i_sm = torch.sigmoid(path_i)
        path_i_att = path_i_sm.unsqueeze(dim=1).repeat(1, self.num_r, 1)

        path_lr = self.fc_lr((res[2] + mm).mean(dim=1))
        path_lr_sm = torch.sigmoid(path_lr)
        path_lr_att = path_lr_sm.unsqueeze(dim=1).repeat(1, self.num_r, 1)

        path_sa = self.fc_sa((res[3] + mm).mean(dim=1))
        path_sa_sm = torch.sigmoid(path_sa)
        path_sa_att = path_sa_sm.unsqueeze(dim=1).repeat(1, self.num_r, 1)  # [bs, num_r, 4]

        path_gr = self.fc_gr((res[4] + mm).mean(dim=1))
        path_gr_sm = torch.sigmoid(path_gr)
        path_gr_att = path_gr_sm.unsqueeze(dim=1).repeat(1, self.num_r, 1)  # [bs, num_r, 4]

        # Dynamic Selection
        next_in = []
        for k in range(self.num_cell):
            feat_ake = path_ake_att[:, :, k].unsqueeze(dim=-1) * res[0]
            feat_i = path_i_att[:, :, k].unsqueeze(dim=-1) * res[1]
            feat_lr = path_lr_att[:, :, k].unsqueeze(dim=-1) * res[2]
            feat_sa = path_sa_att[:, :, k].unsqueeze(dim=-1) * res[3]
            feat_gr = path_gr_att[:, :, k].unsqueeze(dim=-1) * res[4]
            next_in.append(feat_ake + feat_i + feat_sa + feat_gr + feat_lr)

        # Memory Regulation
        mm = self.gru(mm, last_mm)  # Memory
        return (next_in, mm)
