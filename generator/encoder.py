import torch
import re
import numpy as np
import torch.jit as jit
import torch.nn.functional as F

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

class RelationEncoder(nn.Module):
    def __init__(self, rel_embed, vocab, rel_dim, embed_dim, hidden_size, num_layers, dropout):
        super(RelationEncoder, self).__init__()
        self.vocab  = vocab
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rel_embed = rel_embed
        self.rnn = nn.GRU(
            input_size=rel_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout if num_layers > 1 else 0.,
            bidirectional=False
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, src_tokens, src_lengths):
        seq_len, bsz = src_tokens.size()
        sorted_src_lengths, indices = torch.sort(src_lengths, descending=True)
        sorted_src_tokens = src_tokens.index_select(1, indices)
        x = self.rel_embed(sorted_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, sorted_src_lengths.cpu())

        h0 = torch.zeros((self.num_layers, bsz, self.hidden_size), device=\
                x.device, dtype=x.dtype)
        _, final_h = self.rnn(packed_x, h0)

        _, positions = torch.sort(indices)
        final_h = final_h.index_select(1, positions) # num_layers x bsz x hidden_size

        output = self.out_proj(final_h[-1])
        return output

class BagEncoder(jit.ScriptModule):
    def __init__(self, relation_vocab, rel_dim, embed_dim, rel_cnn_dim,
            hidden_size, num_layers, dropout, num_heads, weights_dropout=True, k=4, n_sg=20,
            max_td_depth=40, c=2):
        super(BagEncoder, self).__init__()
        self.c = c
        self.dropout = dropout
        self.rel_embed = AMREmbedding(relation_vocab, rel_dim)
        self.sg_embed = nn.Embedding(n_sg, hidden_size // 4)
        self.bd_embed = nn.Embedding(max_td_depth, hidden_size // 4,
                padding_idx=0)

        self.bs_proj = nn.Linear(rel_dim * k * (k - 1) + self.sg_embed.weight\
                .shape[-1], hidden_size // 2)
        self.nt_size = self.bs_proj.weight.shape[0]
        self.rt_embed = nn.Embedding(1, self.nt_size)
        self.dv_attn_in = MultiheadAttention(self.nt_size, num_heads, dropout,
                weights_dropout=weights_dropout)
        self.dv_attn_out = MultiheadAttention(self.nt_size, num_heads, dropout,
                weights_dropout=weights_dropout)
        self.lvs_proj = nn.Linear(self.nt_size, self.nt_size)
        ch_size = 2 * (self.nt_size + self.bd_embed.weight.shape[-1])
        self.ch_proj_in = nn.Linear(ch_size, self.nt_size)
        self.ch_proj_out = nn.Linear(ch_size, self.nt_size)
        self.rel_proj = nn.Linear(2 * self.nt_size, self.nt_size)
        self.layer_norm = nn.LayerNorm(self.nt_size)

        self.out_proj = nn.Linear(rel_dim + self.nt_size, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.rel_embed.weight, std=.02)
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

    @staticmethod
    def scat(indices, value, sh, base=None, accumulate=False):
        # type: (Tensor, Tensor, List[int], Optional[Tensor], bool) -> Tensor
        if base is None:
            base = torch.zeros(sh, device=value.device, dtype=value.dtype)
        return base.index_put_(indices.split(1, 0), value, accumulate=accumulate)

    def relu_dropout(self, x):
        # type: (Tensor) -> Tensor
        x = F.relu(x, inplace=True)
        return F.dropout(x, p=self.dropout, training=self.training)

    def embed_bags(self, bag_rels, bis, sgs, nt):
        # type: (Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
        # Embed relations per bag
        bsz, b, r = bag_rels.shape
        bs_emb = self.rel_embed(bag_rels.values() - 1)
        bs_emb = self.scat(bag_rels.indices(), bs_emb, torch.Size((bsz, b, r,
            self.rel_embed.weight.shape[-1])))

        # Embed motif per bag
        sg_emb = self.sg_embed(sgs.values().long() - 1)
        sg_emb = self.scat(sgs.indices(), sg_emb, torch.Size((bsz, b,
            self.sg_embed.weight.shape[-1])))
        bs_emb = F.dropout(torch.cat((bs_emb.view(bsz, b, -1), sg_emb), -1),
                p=self.dropout, training=self.training)
        bs_emb = F.relu(self.bs_proj(bs_emb), inplace=True)

        # Select node embeddings from bag ones
        bis_i1, bis_i2 = bis.indices()[0], bis.indices()[1]
        bis_v = bis.values().long() - 1
        bs_emb = self.scat(bis.indices(), bs_emb[bis_i1, bis_v], torch.Size(
            (bsz, nt, bs_emb.shape[-1])))

        bs_emb_r = bs_emb.clone()
        bs_emb_r[:, 0] = self.rt_embed(torch.zeros(1, dtype=torch.long, device=\
                bs_emb_r.device))
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
        dvs = self.scat(dvs_nz, dvs, sh).view(sh[0], sh[1], -1)
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
            bd_emb = self.bd_embed((i - bds[dvs_nz_i, dvs_nz_v]).long())
            dvs = torch.cat((bs_emb[dvs_nz_i, dvs_nz_v], bd_emb), 1)
            bs_emb, dv_weights = self.upd_bs_emb(bs_emb, ns_i1, ns_v, dvs,
                    dvs_nz)
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
        bs_emb_r = self.scat(ind, torch.vstack((a_emb, b_emb)), [ind.shape[1],
            self.nt_size])

        chi_d = (i + 1 - bds_r[chi_i, chi_v]).long()
        bd_emb = self.scat(a_i[None], self.bd_embed(chi_d), [dvs_nz.shape[1],
            self.bd_embed.weight.shape[-1]])
        dvs = torch.cat((bs_emb_r, bd_emb), 1)
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
                self.scat(inds, vals, [-1], base=rel, accumulate=True)
        return rel

    @jit.script_method
    def forward(self, rel_type, d2ns, dts, dv2rels, d2ns_r, dts_r, bag_rels, bis, bds, bds_r, sgs):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
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
        rel0 = self.rel_embed(rel_type)
        bsz, nc = rel_type.shape[2], rel_type.shape[0]
        rel_sh = [bsz, nc, nc, self.nt_size]
        if dv2rels._nnz() > 0:
            bs_emb, bs_emb_r, bis_i1, bis_i2, bis_v = self.embed_bags(
                    bag_rels.coalesce(), bis.coalesce(), sgs.coalesce(),
                    dts.shape[1])  # bsz, nt, h
            bs_emb, ns_ch = self.bottom_up(d2ns, dts, bs_emb, bds)
            bs_emb_r = self.top_down(d2ns_r, dts_r, bs_emb, bs_emb_r, bds_r)

            bs_emb = torch.cat((bs_emb, bs_emb_r), -1)
            bs_emb = self.relu_dropout(self.rel_proj(bs_emb))
            rel = self.collect_rel(ns_ch, bs_emb, dv2rels, rel_sh)
        else:
            rel = torch.zeros(rel_sh, device=rel_type.device)
        rel = self.out_proj(torch.cat((rel0, rel.transpose_(0, 2)), 3))
        return rel

class TokenEncoder(nn.Module):
    def __init__(self, token_vocab, char_vocab, char_dim, token_dim, embed_dim, filters, char2token_dim, dropout, pretrained_file=None):
        super(TokenEncoder, self).__init__()
        self.char_embed = AMREmbedding(char_vocab, char_dim)
        self.token_embed = AMREmbedding(token_vocab, token_dim, pretrained_file)
        self.char2token = CNNEncoder(filters, char_dim, char2token_dim)
        tot_dim = char2token_dim + token_dim
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        self.char_dim = char_dim
        self.token_dim = token_dim
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, token_input, char_input):
        seq_len, bsz, _ = char_input.size()
        char_repr = self.char_embed(char_input.view(seq_len * bsz, -1))
        char_repr = self.char2token(char_repr).view(seq_len, bsz, -1)
        token_repr = self.token_embed(token_input)

        token = F.dropout(torch.cat([char_repr,token_repr], -1), p=self.dropout, training=self.training)
        token = self.out_proj(token)
        return token

class CNNEncoder(nn.Module):
    def __init__(self, filters, input_dim, output_dim, highway_layers=1, d2=False):
        super(CNNEncoder, self).__init__()
        self.d2 = d2
        self.convolutions = nn.ModuleList()
        if d2:
            for width, out_c in filters:
                self.convolutions.append(nn.Conv2d(input_dim, out_c, kernel_size=width))
            self.mp = nn.MaxPool2d(filters[0][0])
        else:
            for width, out_c in filters:
                self.convolutions.append(nn.Conv1d(input_dim, out_c, kernel_size=width))
        final_dim = sum(f[1] for f in filters)
        self.highway = Highway(final_dim, highway_layers)
        self.out_proj = nn.Linear(final_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, input):
        if self.d2:  # input: batch_sz x k x k x input_dim
            x = input.permute(0, 3, 1, 2)
        else:  # input: batch_size x seq_len x input_dim
            x = input.transpose(1, 2)
        conv_result = []
        for i, conv in enumerate(self.convolutions):
            y = conv(x)
            if self.d2:
                if y.shape[-1] > 1:
                    y = self.mp(y)
                y = y.squeeze()
            else:
                y, _ = torch.max(y, -1)
            y = F.relu(y, inplace=True)
            conv_result.append(y)

        conv_result = torch.cat(conv_result, dim=-1)
        conv_result = self.highway(conv_result)
        return self.out_proj(conv_result) #  batch_size x output_dim

class Highway(nn.Module):
    def __init__(self, input_dim, layers):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)

    def forward(self, x):
        for layer in self.layers:
            new_x = layer(x)
            new_x, gate = new_x.chunk(2, dim=-1)
            new_x = F.relu(new_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * new_x
        return x
