import torch
from torch import nn
import torch.nn.functional as F
import math

from encoder import TokenEncoder, BagEncoder
from decoder import DecodeLayer
from graph_transformer import GraphTransformer
from data import ListsToTensor, ListsofStringToTensor, STR

class Generator(nn.Module):
    def __init__(self, concept_dim, rel_dim, nt_size,
            embed_dim, ff_embed_dim, num_heads, dropout,
            graph_layers, pretrained_file, device):
        super(Generator, self).__init__()
        self.bag_encoder = BagEncoder(rel_dim, embed_dim,
                nt_size, dropout, num_heads)
        self.concept_encoder = TokenEncoder(concept_dim, embed_dim,
                self.bag_encoder.nt_size, dropout, pretrained_file)
        self.graph_encoder = GraphTransformer(graph_layers, embed_dim,
                ff_embed_dim, num_heads, dropout)
        self.decoder = DecodeLayer(embed_dim, dropout, device)

        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.token_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.probe_generator = nn.Linear(embed_dim, embed_dim)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.probe_generator.weight, std=0.02)
        nn.init.constant_(self.probe_generator.bias, 0.)

    def con_rel_enc(self, inp):
        relation, root_emb = self.bag_encoder(inp['relation'], inp['d2ns'],
                inp['dts'], inp['dv2rels'], inp['d2ns_r'], inp['dts_r'],
                inp['bag_rels'], inp['bis'], inp['bds'], inp['bds_r'], inp['sgs'])
        concept_repr = self.embed_scale * self.concept_encoder(inp['concept']\
                .coalesce(), root_emb)
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_mask = inp['concept_mask']
        attn_mask = inp['relation_mask']
        bad = attn_mask.all(1)
        if bad.any():
            nz = torch.nonzero(bad)
            attn_mask[nz[:, 0], 0, nz[:, 1]] = 0
        return concept_repr, concept_mask, attn_mask, relation

    def encoder_attn(self, inp):
        with torch.no_grad():
            concept_repr, concept_mask, attn_mask, relation = self.con_rel_enc(inp)
            attn = self.graph_encoder.get_attn_weights(concept_repr, relation,
                    self_padding_mask=concept_mask)
            # nlayers x tgt_len x src_len x bsz x num_heads
        return attn

    def encode_step(self, inp):
        concept_repr, concept_mask, attn_mask, relation = self.con_rel_enc(inp)
        concept_repr = self.graph_encoder(concept_repr, relation,
                self_padding_mask=concept_mask, self_attn_mask=attn_mask)

        probe = torch.tanh(self.probe_generator(concept_repr[:1]))
        return probe

    def forward(self, data):
        probe = self.encode_step(data)
        return self.decoder(probe, data['label'], work=(not self.training))
