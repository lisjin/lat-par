import torch
import re
import numpy as np
import torch.jit as jit
import torch.nn.functional as F

from mol_encoder import AtomEncoder, BondEncoder  # copied from ogb.graphproppred
from torch import nn, Tensor
from typing import List, Tuple
from torch.nn.parameter import Parameter

from transformer import Embedding, MultiheadAttention

def AMREmbedding(vocab, embedding_dim, pretrained_file=None, amr=False, dump_file=None):
    if pretrained_file is None:
        return Embedding(vocab.size, embedding_dim, vocab.padding_idx)

    tokens_to_keep = set()
    for idx in range(vocab.size):
        token = vocab.idx2token(idx)
        # TODO: Is there a better way to do this? Currently we have a very specific 'amr' param.
        if amr:
            token = re.sub(r'-\d\d$', '', token)
        tokens_to_keep.add(token)

    embeddings = {}

    if dump_file is not None:
        fo = open(dump_file, 'w', encoding='utf8')

    with open(pretrained_file, encoding='utf8') as embeddings_file:
        for line in embeddings_file.readlines():
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                continue
            token = fields[0]
            if token in tokens_to_keep:
                if dump_file is not None:
                    fo.write(line)
                vector = np.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector

    if dump_file is not None:
        fo.close()

    all_embeddings = np.asarray(list(embeddings.values()))
    print ('pretrained', all_embeddings.shape)
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    embedding_matrix = torch.FloatTensor(vocab.size, embedding_dim).normal_(embeddings_mean,
                                                                            embeddings_std)

    for i in range(vocab.size):
        token = vocab.idx2token(i)

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
        else:
            if amr:
                normalized_token = re.sub(r'-\d\d$', '', token)
                if normalized_token in embeddings:
                    embedding_matrix[i] = torch.FloatTensor(embeddings[normalized_token])
    embedding_matrix[vocab.padding_idx].fill_(0.)

    return nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

def sp_scat(indices, value, sh, base=None, accumulate=False):
    # type: (Tensor, Tensor, List[int], Optional[Tensor], bool) -> Tensor
    if base is None:
        base = torch.zeros(sh, device=value.device, dtype=value.dtype)
    return base.index_put_(indices.split(1, 0), value, accumulate=accumulate)

class BagEncoder(jit.ScriptModule):
    def __init__(self, rel_dim, embed_dim, nt_size,
            dropout, num_heads, weights_dropout=True, k=4, n_sg=15,
            max_td_depth=49, c=2):
        super(BagEncoder, self).__init__()
        self.c = c
        self.dropout = dropout
        self.rel_embed = BondEncoder(emb_dim=rel_dim)
        self.rl_embed = nn.Embedding(3, rel_dim)  # self-loop, cls, cls_rev
        self.sg_embed = nn.Embedding(n_sg, nt_size)# // 2)
        self.bd_embed = nn.Embedding(max_td_depth, nt_size, padding_idx=0)# // 2, padding_idx=0)
        self.bs_proj = nn.Linear(rel_dim * k * (k - 1) + self.sg_embed.weight\
                .shape[-1], nt_size)

        self.nt_size = nt_size
        self.rt_embed = nn.Embedding(1, nt_size)
        self.dv_attn_in = MultiheadAttention(nt_size, num_heads, dropout,
                weights_dropout=weights_dropout)
        self.dv_attn_out = MultiheadAttention(nt_size, num_heads, dropout,
                weights_dropout=weights_dropout)
        self.lvs_proj = nn.Linear(nt_size, nt_size)

        ch_size = 2 * (nt_size + self.bd_embed.weight.shape[-1])
        #ch_size = 2 * nt_size
        self.ch_proj_in = nn.Linear(ch_size, nt_size)
        self.ch_proj_out = nn.Linear(ch_size, nt_size)
        self.rel_proj = nn.Linear(2 * nt_size, nt_size)
        self.layer_norm = nn.LayerNorm(nt_size)
        self.out_proj = nn.Linear(rel_dim + nt_size, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rl_embed.weight.data)
        nn.init.normal_(self.sg_embed.weight, std=.02)
        nn.init.normal_(self.bd_embed.weight, std=.02)
        nn.init.normal_(self.rt_embed.weight, std=.02)

        nn.init.normal_(self.bs_proj.weight, std=.02)
        nn.init.constant_(self.bs_proj.bias, 0.)
        nn.init.normal_(self.lvs_proj.weight, std=.02)
        nn.init.constant_(self.lvs_proj.bias, 0.)
        nn.init.normal_(self.ch_proj_in.weight, std=.02)
        nn.init.constant_(self.ch_proj_in.bias, 0.)
        nn.init.normal_(self.ch_proj_out.weight, std=.02)
        nn.init.constant_(self.ch_proj_out.bias, 0.)
        nn.init.normal_(self.out_proj.weight, std=.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def relu_dropout(self, x):
        # type: (Tensor) -> Tensor
        x = F.relu(x, inplace=True)
        return F.dropout(x, p=self.dropout, training=self.training)

    def embed_bags(self, bag_rels, bis, sgs, nt):
        # type: (Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        # Embed relations per bag
        bsz, b, r, _ = bag_rels.shape
        bs_emb = self.rel_embed(bag_rels.values() - 1)
        bs_emb = F.dropout(bs_emb, p=self.dropout, training=self.training)
        bs_emb = sp_scat(bag_rels.indices(), bs_emb, torch.Size((bsz, b, r,
            bs_emb.shape[-1])))

        # Embed motif per bag
        sg_emb = self.sg_embed(torch.clamp(sgs.values().long() - 1, 0,
                self.sg_embed.num_embeddings - 1))
        sg_emb = sp_scat(sgs.indices(), sg_emb, torch.Size((bsz, b,
            self.sg_embed.weight.shape[-1])))
        bs_emb = F.dropout(torch.cat((bs_emb.view(bsz, b, -1), sg_emb), -1),
                p=self.dropout, training=self.training)
        bs_emb = F.relu(self.bs_proj(bs_emb), inplace=True)

        # Select node embeddings from bag ones
        bis_i1, bis_i2 = bis.indices()[0], bis.indices()[1]
        bis_v = bis.values().long() - 1
        bs_emb = sp_scat(bis.indices(), bs_emb[bis_i1, bis_v], torch.Size(
            (bsz, nt, bs_emb.shape[-1])))

        bs_emb_r = bs_emb.clone()
        bs_emb_r[:, 0] = self.rt_embed(torch.zeros(1, dtype=torch.long,
            device=bs_emb_r.device))
        return bs_emb, bs_emb_r, bis_i1, bis_i2, bis_v

    @staticmethod
    def get_dvs_nz(ns, dts):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        # Collect children node IDs, parent/child depths
        ns_i1, ns_v = ns.indices()[0], ns.values().long() - 1
        dvs = dts[ns_i1, ns_v]  # ?, v, 2
        dvs_nz = torch.nonzero(dvs).t_()
        dvs_nz_v = dvs[dvs_nz[0], dvs_nz[1], dvs_nz[2]].long() - 1
        dvs_nz_i = ns_i1[dvs_nz[0]]  # batch indices
        return dvs, ns_i1, ns_v, dvs_nz, dvs_nz_i, dvs_nz_v

    def upd_bs_emb(self, bs_emb, ns_i1, ns_v, dvs, dvs_nz, need_weights=True):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tuple[Tensor, Tensor]
        # Flatten sibling nodes then combine
        sh = [ns_i1.shape[0], int(dvs_nz[1].max()) + 1, self.c, dvs.shape[1]]
        dvs = sp_scat(dvs_nz, dvs, sh).view(sh[0], sh[1], -1)
        dvs = F.relu(self.ch_proj_in(dvs) if need_weights else self.ch_proj_out(\
                dvs), inplace=True)

        # Per-node attention over derivations
        key_padding_mask = torch.ones(dvs.shape[:-1], device=dvs.device, dtype=\
                torch.bool).transpose(0, 1)
        key_padding_mask[dvs_nz[1], dvs_nz[0]] = 0
        par = bs_emb[ns_i1, ns_v].unsqueeze(0)
        if need_weights:
            par, dv_weights = self.dv_attn_in(query=par, key=dvs.transpose(0, 1),
                    key_padding_mask=key_padding_mask, need_weights=need_weights)
            dv_weights = dv_weights.squeeze(0).softmax(1)
        else:
            par, _ = self.dv_attn_out(query=par, key=dvs.transpose(0, 1),
                    key_padding_mask=key_padding_mask, need_weights=need_weights)
            dv_weights = torch.randn(0, 2)
        par = F.dropout(par.squeeze(0), p=self.dropout, training=self.training)

        # Scatter parents back to node embeddings
        bs_emb[ns_i1, ns_v] = self.layer_norm(bs_emb[ns_i1, ns_v] + par)
        return bs_emb, dv_weights

    def leaf_trans(self, dts, bs_emb, lvs):
        # type: (Tensor, Tensor, Tensor) -> Tensor
        lvs = lvs.coalesce()
        ns_i1, ns_v = lvs.indices()[0], lvs.values().long() - 1
        bs_emb[ns_i1, ns_v] = self.relu_dropout(self.lvs_proj(bs_emb[ns_i1, ns_v]))
        return bs_emb

    def bottom_up(self, d2ns, dts, bs_emb, bds):
        # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, List[Tuple[Tensor, Tensor, Tensor, Tensor]]]
        ns_lst = d2ns.unbind(0)
        bs_emb = self.leaf_trans(dts, bs_emb, ns_lst[0])
        ns_ch = jit.annotate(List[Tuple[Tensor, Tensor, Tensor, Tensor]], [])
        for i in range(1, len(ns_lst)):
            dvs, ns_i1, ns_v, dvs_nz, dvs_nz_i, dvs_nz_v = self.get_dvs_nz(
                    ns_lst[i].coalesce(), dts)
            bd_emb = torch.clamp((i - bds[dvs_nz_i, dvs_nz_v]).long(), 0,\
                    self.bd_embed.num_embeddings - 1)
            bd_emb = self.bd_embed(bd_emb)
            dvs = torch.cat((bs_emb[dvs_nz_i, dvs_nz_v], bd_emb), 1)
            #dvs = bs_emb[dvs_nz_i, dvs_nz_v]
            bs_emb, dv_weights = self.upd_bs_emb(bs_emb, ns_i1, ns_v, dvs, dvs_nz)
            ns_ch.append((dvs_nz_i, dvs_nz_v, dv_weights[dvs_nz[0], dvs_nz[1]],
                ns_v[dvs_nz[0]]))
        return bs_emb, ns_ch

    @staticmethod
    def filt_bs(bs_emb, dvs_nz2, dvs_nz_i, dvs_nz_v, tgt):
        # type: (Tensor, Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor]
        ind = torch.nonzero(dvs_nz2 == tgt)[:, 0]
        chi_i, chi_v = dvs_nz_i[ind], dvs_nz_v[ind]
        return ind, bs_emb[chi_i, chi_v], chi_i, chi_v

    def ch_embed_r(self, bs_emb, bs_emb_r, bds_r, dvs_nz, dvs_nz_i, dvs_nz_v,
            ns_i1, i, n_chi):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, int) -> Tensor
        # a: outside parents, b: inside siblings
        a_i, a_emb, chi_i, chi_v = self.filt_bs(bs_emb_r, dvs_nz[2], dvs_nz_i,
                dvs_nz_v, 0)
        b_i, b_emb, _, _ = self.filt_bs(bs_emb, dvs_nz[2], dvs_nz_i, dvs_nz_v, 1)
        ind = torch.hstack((a_i[None], b_i[None]))
        bs_emb_r = sp_scat(ind, torch.vstack((a_emb, b_emb)), [ind.shape[1],
            self.nt_size])

        chi_d = (i + 1 - bds_r[chi_i, chi_v]).long()
        bd_emb = sp_scat(a_i[None], self.bd_embed(chi_d), [dvs_nz.shape[1],
            self.bd_embed.weight.shape[-1]])
        dvs = torch.cat((bs_emb_r, bd_emb), 1)
        #dvs = bs_emb_r
        return dvs

    def top_down(self, d2ns, dts, bs_emb, bs_emb_r, bds):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        ns_lst = d2ns.unbind(0)[1:]
        for i in range(len(ns_lst)):
            dvs, ns_i1, ns_v, dvs_nz, dvs_nz_i, dvs_nz_v = self.get_dvs_nz(
                    ns_lst[i].coalesce(), dts)
            dvs = self.ch_embed_r(bs_emb, bs_emb_r, bds, dvs_nz, dvs_nz_i,
                    dvs_nz_v, ns_i1, i, dvs.shape[0])
            bs_emb_r, _ = self.upd_bs_emb(bs_emb_r, ns_i1, ns_v, dvs, dvs_nz,
                    need_weights=False)
        return bs_emb_r

    def collect_rel(self, ns_ch, bs_emb, dv2rels, rel_sh):
        # type: (List[Tuple[Tensor, Tensor, Tensor, Tensor]], Tensor, Tensor, List[int]) -> Tensor
        scores = torch.zeros(bs_emb.shape[:-1], dtype=torch.float32, device=bs_emb.device)
        scores[:, 0] = 1.
        rel = torch.zeros(rel_sh, device=bs_emb.device)
        d_lst = [dr.coalesce() for dr in dv2rels.unbind(0)[1:]]
        for i in range(len(ns_ch) - 1, -1, -1):
            c1, c2, c3, c4 = ns_ch[i]
            msg = c3 * scores[c1, c4]
            scores[c1, c2] += msg
            if i < len(d_lst) and d_lst[i]._nnz():
                ic = d_lst[i].indices()[0]
                c1, c4 = c1[ic], c4[ic]
                inds = torch.vstack((c1, d_lst[i].values().long().T))
                vals = bs_emb[c1, c4] * msg[ic, None]
                sp_scat(inds, vals, [-1], base=rel, accumulate=True)
        return rel

    def get_rel0(self, rel_type, bsz, nc):
        # type: (Tensor, int, int) -> Tensor
        rel0 = self.rel_embed(rel_type.values())
        rel0 = F.dropout(rel0, p=self.dropout, training=self.training)
        rel0 = sp_scat(rel_type.indices(), rel0, torch.Size((nc, nc, bsz,
            rel0.shape[-1])))
        ix = torch.arange(nc)
        rel0[ix, ix] = self.rl_embed(torch.tensor(0, device=rel0.device))
        rel0[0, 1:] = self.rl_embed(torch.tensor(1, device=rel0.device))
        rel0[1:, 0] = self.rl_embed(torch.tensor(2, device=rel0.device))
        return rel0

    @jit.script_method
    def forward(self, rel_type, d2ns, dts, dv2rels, d2ns_r, dts_r, bag_rels, bis, bds, bds_r, sgs):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """Initialize nodes with relevant bag embeddings, then perform bottom-up
        computation over forest. Propagate node embeddings back to relations.
        d2ns:       d, bsz, u           (sparse)
        dts:        bsz, nt, v, 2
        dv2rels:    d, bsz, nt, n_rel     (sparse)
        bag_rels:   bsz, b, k(k - 1)    (sparse)
        bis:        bsz, nt2            (sparse)
        bds:        bsz, nt
        sgs:        bsz, nt2            (sparse)
        """
        bsz, nc = rel_type.shape[2], rel_type.shape[0]
        rel0 = self.get_rel0(rel_type.coalesce(), bsz, nc)
        rel_sh = [bsz, nc, nc, self.nt_size]
        root_emb = torch.zeros((bsz, self.nt_size), device=rel0.device)
        if dv2rels._nnz() > 0:
            bs_emb, bs_emb_r, bis_i1, bis_i2, bis_v = self.embed_bags(
                    bag_rels.coalesce(), bis.coalesce(), sgs.coalesce(),
                    dts.shape[1])  # bsz, nt, h
            bs_emb, ns_ch = self.bottom_up(d2ns, dts, bs_emb, bds)
            root_emb[:] = bs_emb[:, 0]
            bs_emb_r = self.top_down(d2ns_r, dts_r, bs_emb, bs_emb_r, bds_r)

            bs_emb = torch.cat((bs_emb, bs_emb_r), -1)
            bs_emb = self.relu_dropout(self.rel_proj(bs_emb))
            rel = self.collect_rel(ns_ch, bs_emb, dv2rels, rel_sh)
        else:
            rel = torch.zeros(rel_sh, device=rel_type.device)
        rel = self.out_proj(torch.cat((rel0, rel.transpose_(0, 2)), 3))
        return rel, root_emb

class TokenEncoder(jit.ScriptModule):
    def __init__(self, token_dim, embed_dim, nt_size, dropout, pretrained_file=None):
        super(TokenEncoder, self).__init__()
        self.token_embed = AtomEncoder(emb_dim=token_dim)
        self.out_proj = nn.Linear(token_dim, embed_dim)
        self.token_dim = token_dim
        self.root_proj = nn.Linear(nt_size, embed_dim)
        self.dropout = dropout
        self.glbl_embed = nn.Embedding(1, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.normal_(self.root_proj.weight, std=0.02)
        nn.init.constant_(self.root_proj.bias, 0.)
        nn.init.normal_(self.glbl_embed.weight, std=.02)

    @jit.script_method
    def forward(self, token_input, root_emb):
        nv, bsz, _ = token_input.shape
        token_repr = self.token_embed(token_input.values())
        token_repr = sp_scat(token_input.indices(), token_repr, torch.Size((nv,
            bsz, token_repr.shape[-1])))
        token_repr = self.out_proj(token_repr)
        token_repr[0] = self.glbl_embed(torch.tensor(0, device=token_repr.device))
        #token_repr[0] = self.root_proj(root_emb)
        token_repr = F.dropout(token_repr, p=self.dropout, training=self.training)
        return token_repr

