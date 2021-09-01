import torch
from torch import nn
import torch.nn.functional as F

class DecodeLayer(nn.Module):
    def __init__(self, embed_dim, dropout, device):
        super(DecodeLayer, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(embed_dim, 1)
        self.reset_parameters(device)

    def reset_parameters(self, device):
        nn.init.normal_(self.fc.weight, std=.02)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, probe, target=None, work=False):
        # probe: tgt_len x bsz x embed_dim
        probe = F.dropout(probe, p=self.dropout, training=self.training)
        inp = self.fc(probe.squeeze(0))
        loss = nn.BCEWithLogitsLoss()(inp, target.float().unsqueeze(-1))
        return loss, inp, target
