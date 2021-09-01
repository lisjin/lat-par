import torch
import math

import torch.jit as jit
import torch.nn.functional as F

from torch import nn, Tensor
from torch.nn import Parameter
from typing import List

class GraphTransformer(jit.ScriptModule):
    def __init__(self, layers, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(GraphTransformerLayer(embed_dim, ff_embed_dim,
                num_heads, dropout, weights_dropout))

    @jit.script_method
    def forward(self, x, relation, kv = None, self_padding_mask = None,
            self_attn_mask = None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        """td_masks: b x d x tgt_len, src_len"""
        for i, layer in enumerate(self.layers):
            x, _ = layer(x, relation, kv=kv, self_padding_mask=self_padding_mask,
                    self_attn_mask=self_attn_mask)
        return x

    def get_attn_weights(self, x, relation, kv = None,
                self_padding_mask = None, self_attn_mask = None):
        attns = []
        for idx, layer in enumerate(self.layers):
            _, attn = layer(x, relation, kv, self_padding_mask, self_attn_mask)
            attns.append(attn)
        attn = torch.stack(attns)
        return attn

class GraphTransformerLayer(jit.ScriptModule):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformerLayer, self).__init__()
        self.self_attn = RelationMultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.fc1 = nn.Linear(embed_dim, ff_embed_dim)
        self.fc2 = nn.Linear(ff_embed_dim, embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    @jit.script_method
    def forward(self, x, relation, kv=None, self_padding_mask=None,
            self_attn_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(x, relation=relation, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, relation=relation, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn

class RelationMultiheadAttention(jit.ScriptModule):
    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(RelationMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.relation_in_proj = nn.Linear(embed_dim, 2*embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.relation_in_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    @jit.script_method
    def forward(self, query, relation, key=None, value=None, key_padding_mask=None, attn_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        """ Input shape: Time x Batch x Channel
            relation:  tgt_len x src_len x bsz x dim
            key_padding_mask: src_len x batch
            attn_mask:  tgt_len x src_len
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = relation.shape[1]

        if key is None:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif value is None:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        ra, rb = self.relation_in_proj(relation).chunk(2, dim=-1)
        ra = ra.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        rb = rb.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q.unsqueeze(1) + ra
        k = k.unsqueeze(0) + rb
        q *= self.scaling
        # q: tgt_len x src_len x bsz*heads x dim
        # k: tgt_len x src_len x bsz*heads x dim
        # v: src_len x bsz*heads x dim

        attn_weights = torch.einsum('ijbn,ijbn->ijb', [q, k])
        assert list(attn_weights.size()) == [tgt_len, src_len, bsz * self.num_heads]

        if attn_mask is not None:
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(-1),
                float('-inf')
            )
            attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)


        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(0).unsqueeze(-1),
                float('-inf')
            )
            attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)

        attn_weights = F.softmax(attn_weights, dim=1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights: tgt_len x src_len x bsz*heads
        # v: src_len x bsz*heads x dim
        attn = torch.einsum('ijb,jbn->bin', [attn_weights, v])
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # maximum attention weight over heads
        attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=-1):
        # type: (Tensor, int, int) -> Tensor
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end < 0:
            end = bias.shape[0]
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
