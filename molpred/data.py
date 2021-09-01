import h5py
import marshal
import random
import torch
import numpy as np
import os
import json
import _pickle as pickle

from itertools import product, zip_longest
from ogb.graphproppred import DglGraphPropPredDataset
from torch.utils.data import IterableDataset, get_worker_info

from AMRGraph import Molecule

PAD, UNK = '<PAD>', '<UNK>'
CLS = '<CLS>'
STR, END = '<STR>', '<END>'
SEL, rCLS = '<SELF>', '<rCLS>'

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        num_tot_tokens = 0
        num_vocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.strip().split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                print(line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_vocab_tokens += cnt
            self._priority[token] = int(cnt)
        self.coverage = num_vocab_tokens/num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        return self._priority.get(x, 0)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

def _back_to_txt_for_check(tensor, vocab, local_idx2token=None):
    for bid, xs in enumerate(tensor.t().tolist()):
        txt = []
        for x in xs:
            if x == vocab.padding_idx:
                break
            if x >= vocab.size:
                assert local_idx2token is not None
                assert local_idx2token[bid] is not None
                tok = local_idx2token[bid][x]
            else:
                tok = vocab.idx2token(x)
            txt.append(tok)
        txt = ' '.join(txt)
        print (txt)

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    end_all = len(dataset.data)
    per_worker = end_all // worker_info.num_workers
    start = worker_info.id * per_worker
    end = min(start + per_worker, end_all)
    dataset.data = dataset.data[start:end]

def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0., pad=None, dtype=torch.long):
    if pad is None:
        pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = torch.tensor(ys, dtype=dtype).t_().contiguous()
    return data

def ListsofStringToTensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([STR]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data

def ArraysToTensor(xs, fill_val=0):
    """Return list of numpy arrays of the same dimensionality"""
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.full(shape, fill_val, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
        tensor = torch.from_numpy(data).long()
    return tensor

def add_forest_tensors(d, d2j, k, max_nn=1e4):
    def build_dt(gr, pr):
        """Build original forest dict then binarize.
        key: index within `pr`
        value: list of lists, where sublist is deriv or `pr` indices of children
        """
        dt = {}
        for par, chi in gr:
            dt.setdefault(par, {}).setdefault(pr[chi, 0], []).append(chi)

        to_fix = []
        for k in dt.keys():
            dt[k] = list(dt[k].values())
            to_fix += [(k, k2) for k2, v2 in enumerate(dt[k]) if len(v2) > 2]

        def bal_bin(dt, lst, n2par, par, new_i):
            if len(lst) == 2:
                return new_i
            out = []
            for j in range(0, len(lst), 2):
                if j != len(lst) - 1:
                    dt[new_i] = [[lst[j], lst[j + 1]]]
                    out.append(new_i)
                    n2par[new_i] = par
                    new_i += 1
                else:
                    out.append(lst[-1])
            lst[:] = out
            return bal_bin(dt, lst, n2par, par, new_i)

        n2par = {}
        new_i = pr.shape[0]
        for t in to_fix:
            new_i = bal_bin(dt, dt[t[0]][t[1]], n2par, t[0], new_i)

        dt_r = {}
        for par in dt.keys():
            for v2 in dt[par]:
                dt_r.setdefault(v2[0], []).append([par])
                if len(v2) > 1:
                    dt_r[v2[0]][-1].append(v2[0])
                    dt_r.setdefault(v2[1], []).append([par, v2[0]])
                    dt_r.setdefault(v2[0], []).append([par, v2[1]])
        return dt, dt_r, n2par, new_i

    def get_depth(dt, n2d, root, rev=False):
        """Recompute all node depths."""
        max_d_c = 0
        if n2d[root] < 0:
            if root in dt:
                max_d = 0
                for dv in dt[root]:
                    if rev:
                        max_d = max(max_d, get_depth(dt, n2d, dv[0], rev=rev)[0])
                    else:
                        max_d = max(max_d, max(get_depth(dt, n2d, chi, rev=rev)[0] for\
                            chi in dv))
                n2d[root] = max_d + 1
                max_d_c = max(max_d_c, n2d[root])
            else:
                n2d[root] = 0
        return n2d[root], max_d_c

    def offset_frag(sep2frag, si):
        return (v + 1 for v in sep2frag[si][1:])

    def collect_bags(efeat, e2i, sep2frag, n_cur):
        sg = []
        bag_rel = []
        bag_v = []
        b2bi = {sep_id: j for j, sep_id in enumerate(sep2frag.keys())}
        bi = [-1] + [b2bi[sep_id] for sep_id in pr[1:, 0]]
        rel2len = {}
        for sep_id, j in b2bi.items():
            sg.append(sep2frag[sep_id][0])
            bag = list(offset_frag(sep2frag, sep_id))
            ix = np.ix_(bag, bag)
            br = [efeat[e2i[(t[0], t[1])]] for t in product(bag, repeat=2) if\
                    (t[0], t[1]) in e2i]
            bag_rel.append(br if br else None)
            bag_v.append(bag)
        return bi, sg, bag_rel, bag_v

    def dt2mats(dt, n2par, n2d, max_d_a, pr, bi, fill_val=-1, b_up=False, e2i=None):
        """Compress forest to matrices."""
        m1 = [[] for _ in range(max_d_a)]  # node indices per depth
        max_u = 0
        max_d = 0
        for n, d in enumerate(n2d):
            m1[d].append(n)
            max_d = max(max_d, d)
        for d in range(max_d + 1):
            max_u = max(max_u, len(m1[d]))

        def get_v_set(b_id):
            return frozenset(bag[bi[b_id]]) if b_id > 0 else frozenset()

        def chi_rels(chi, par):
            out = set()
            if chi < len(pr):
                chi_v = get_v_set(chi)
                par_v = get_v_set(par)
                xv = par_v & chi_v if par_v else set()
                out |= set(t for t in product(chi_v, repeat=2) if t[0] != t[1]\
                        and not (t[0] in xv and t[1] in xv) and (t[0], t[1]) in\
                        e2i)
            return out

        m2 = []  # derivations per node
        max_nl, max_nd, max_v_c = (0,) * 3
        dv2rel = {}  # keyed by (depth, j, dv_i)
        dt_ks = set(dt.keys())
        for node in range(len(n2d)):
            if node in dt_ks:
                max_v_c = max(max_v_c, len(dt[node]))
                max_nd = max(max_nd, n2d[node])
                cur_d = n2d[node]
                if bag is not None:
                    par = n2par[node] if node >= len(pr) else node
                m2.append([])
                for h, l in enumerate(dt[node]):
                    m2[-1].extend([l + [fill_val] * (2 - len(l))])
                    if b_up:
                        d2j.setdefault(cur_d, 0)
                        m = 0
                        for chi in l:
                            chi = n2par[chi] if chi >= len(pr) else chi
                            for o in chi_rels(chi, par):
                                dv2rel[(cur_d, d2j[cur_d], m)] = o
                                m += 1
                            d2j[cur_d] += 1
            else:
                m2.append(None)
        return m1, m2, dv2rel, max_nl, max_v_c, max_u

    max_d_a, max_d_a_r = (1,) * 2
    pr = d['pr'].reshape(-1, 3)
    gr = d['gr'].reshape(-1, 2)
    dt, dt_r, n2par, nv = build_dt(gr, pr)
    if 0 < len(pr) and len(pr) < max_nn:
        n2d = [-1] * nv
        _, max_d_c = get_depth(dt, n2d, 0)
        max_d_a = max(max_d_a, max_d_c + 1)
        n2d_r = [-1] * nv
        for root in set(range(nv)) - set(dt.keys()):  # leaves
            _, max_d_c_r = get_depth(dt_r, n2d_r, root, rev=True)
            max_d_a_r = max(max_d_a_r, max_d_c_r + 1)
        assert(min(n2d) + min(n2d_r) == 0)

        n_cur = d['nfeat'].shape[0] + 1
        bi, sg, bag_rel, bag = collect_bags(d['efeat'], d['e2i'], d['sep2frag'], n_cur)
        d2n, dt, dv2rel, d['max_nl'], d['max_v'], d['max_u'] = dt2mats(dt,
                n2par, n2d, max_d_a, pr, bi, b_up=True, e2i=d['e2i'])
        d2n_r, dt_r, _, d['max_nl_r'], d['max_v_r'], d['max_u_r'] = dt2mats(
                dt_r, n2par, n2d_r, max_d_a_r, pr, bi)
    else:
        d2n, dt, dv2rel, d2n_r, dt_r, bag_rel, bi, n2d, n2d_r, sg = (None,) * 10
    return d2n, dt, dv2rel, d2n_r, dt_r, bag_rel, bi, n2d, n2d_r, sg

def sparsify_batch(batch, sp_keys=('d2ns', 'd2ns_r', 'bag_rels', 'dv2rels',
    'bis', 'sgs', 'concept', 'relation')):
    for sk in sp_keys:
        batch[sk] = torch.sparse_coo_tensor(batch[sk][0], batch[sk][1],
                batch[sk][2])

def short_sparse(inds, vals, shape=None):
    inds = torch.tensor(np.stack(inds).T)
    vals = torch.tensor(vals)
    return torch.sparse_coo_tensor(inds, vals, shape) if shape is not None else\
            torch.sparse.ShortTensor(inds, vals)

def custom_sparse(sp):
    sp = sp.coalesce()
    return sp.indices(), sp.values(), sp.shape

def get_forest_mats(data, n_conc, ef_dim, k=4):
    def dcts2mat(dcts, dtype=torch.short):
        inds = []
        vals = []
        for dct in dcts:
            if dct:
                inds.extend(list(dct.keys()))
                vals.extend(list(dct.values()))
        if len(inds):
            m = short_sparse(inds, vals)
        else:
            m = torch.zeros((0, 2), dtype=dtype).to_sparse()
        return custom_sparse(m)

    def to_maybe_sparse(m, val, sp, keep_last=False):
        m = torch.from_numpy(m) + int(val < 0)
        if sp:
            sparse_dims = len(m.shape) - int(keep_last)
            m = custom_sparse(m.to_sparse(sparse_dims))
        return m

    def lsts2mat_3d(lsts, shape, dtype=np.short, val=-1, sp=True, tp=False,
            keep_last=False):
        m = np.full(shape, val, dtype=dtype)
        for i, l in enumerate(lsts):
            if l is not None:
                for j, l2 in enumerate(l):
                    if l2 is not None:
                        m[i, j, :len(l2)] = l2
        if tp:
            m = m.swapaxes(0, 1)
        return to_maybe_sparse(m, val, sp, keep_last=keep_last)

    def lsts2mat_2d(lsts, shape, dtype=np.short, val=0, sp=True):
        m = np.full(shape, val, dtype=dtype)
        for i, l in enumerate(lsts):
            if l is not None:
                m[i][:len(l)] = l
        return to_maybe_sparse(m, val, sp)

    def upd_maxes(d2n, max_nl_c, max_nl, max_v_c, max_d_a, max_v_a, max_u_c, max_u):
        max_d_a = max(max_d_a, len(d2n))
        max_v_a = max(max_v_a, max_v_c)
        max_nl = max(max_nl, max_nl_c)
        max_u = max(max_u, max_u_c)
        return max_d_a, max_v_a, max_nl, max_u

    max_n_a, max_n2_a, max_d_a, max_d_a_r, max_u_a, max_u_a_r, max_b_a, max_nl,\
            max_nl_r, max_v_a, max_v_a_r = (1,) * 11
    d2ns, dts, dv2rels, d2ns_r, dts_r, bag_rels, bis, bds, bds_r, sgs = ([] for _ in range(10))
    d2j = {}
    for i, d in enumerate(data):
        d2n, dt, dv2rel, d2n_r, dt_r, bag_rel, bi, n2d, n2d_r, sg =\
                add_forest_tensors(d, d2j, k)

        if 'max_v' in d:
            max_d_a, max_v_a, max_nl, max_u_a = upd_maxes(d2n, d['max_nl'],
                    max_nl, d['max_v'], max_d_a, max_v_a, d['max_u'], max_u_a)
            max_d_a_r, max_v_a_r, max_nl_r, max_u_a_r = upd_maxes(d2n_r,
                    d['max_nl_r'], max_nl_r, d['max_v_r'], max_d_a_r, max_v_a_r,
                    d['max_u_r'], max_u_a_r)
            max_n_a = max(max_n_a, len(n2d))
            max_n2_a = max(max_n2_a, len(bi))
            max_b_a = max(max_b_a, len(bag_rel))
        d2ns.append(d2n)
        dts.append(dt)
        d2ns_r.append(d2n_r)
        dts_r.append(dt_r)
        bag_rels.append(bag_rel)
        dv2rels.append(dv2rel)
        bis.append(bi)
        bds.append(n2d)
        bds_r.append(n2d_r)
        sgs.append(sg)

    bsz = len(data)
    d2ns = lsts2mat_3d(d2ns, (bsz, max_d_a, max_u_a), tp=True)
    dts = lsts2mat_3d(dts, (bsz, max_n_a, max_v_a, 2), sp=False)
    d2ns_r = lsts2mat_3d(d2ns_r, (bsz, max_d_a_r, max_u_a_r), tp=True)
    dts_r = lsts2mat_3d(dts_r, (bsz, max_n_a, max_v_a_r, 2), sp=False)
    bag_rels = lsts2mat_3d(bag_rels, (bsz, max_b_a, k * (k - 1), ef_dim), dtype=np.int, keep_last=True)

    dv2rels = dcts2mat(dv2rels)

    bis = lsts2mat_2d(bis, (bsz, max_n2_a), val=-1)
    bds = lsts2mat_2d(bds, (bsz, max_n_a), sp=False)
    bds_r = lsts2mat_2d(bds_r, (bsz, max_n_a), sp=False)
    sgs = lsts2mat_2d(sgs, (bsz, max_b_a), val=-1)
    return d2ns, dts, dv2rels, d2ns_r, dts_r, bag_rels, bis, bds, bds_r, sgs

def batchify(data, is_train=True, nf_dim=9, ef_dim=3):
    c0 = np.full((1, nf_dim), -1, dtype=data[0]['nfeat'].dtype)  # global vertex
    _conc = ArraysToTensor([np.vstack((c0, x['nfeat'])) for x in data], fill_val=-1).transpose_(0, 1).contiguous()
    _conc_mask = _conc[:, :, 0] > -1
    _ci = _conc_mask.nonzero()
    _cv = _conc[_ci[:, 0], _ci[:, 1]]
    _conc = custom_sparse(short_sparse(_ci, _cv))
    _conc_mask = torch.logical_not(_conc_mask)
    nv, bsz = _conc[2][:2]
    rel_sh = (bsz, nv, nv)

    # x['e2i'].keys() are offset by one to account for global vertex 0
    rel_inds = np.vstack([np.hstack((np.full((len(x['e2i']), 1), i),
        np.stack(list(x['e2i'].keys())) + 1)) for i, x in enumerate(data)])
    rel_vals = np.vstack([x['efeat'] for x in data])
    _relation_type = custom_sparse(short_sparse(rel_inds, rel_vals, shape=(*rel_sh, rel_vals.shape[-1])).transpose_(0, 2))  # [tgt, src, bsz, ef_dim]

    rel_out_inds = np.vstack([np.hstack((np.full((x['e_out'].shape[0], 1), i),
        x['e_out'] + 1)) for i, x in enumerate(data) if x['e_out'] is not None])
    _relation_mask = short_sparse(rel_out_inds, np.ones(rel_out_inds.shape[0]),
            shape=rel_sh).transpose_(0, 2).to_dense().bool()

    d2ns, dts, dv2rels, d2ns_r, dts_r, bag_rels, bis, bds, bds_r, sgs =\
            get_forest_mats(data, _conc[2][0], ef_dim)
    label = torch.LongTensor([x['label'] for x in data])

    ret = {
        'concept': _conc,
        'concept_mask': _conc_mask,
        'relation': _relation_type,
        'relation_mask': _relation_mask,
        'd2ns': d2ns,
        'dts': dts,
        'dv2rels': dv2rels,
        'd2ns_r': d2ns_r,
        'dts_r': dts_r,
        'bag_rels': bag_rels,
        'bis': bis,
        'bds': bds,
        'bds_r': bds_r,
        'sgs': sgs,
        'label': label
    }
    return ret

class MyDataset(IterableDataset):
    def __init__(self, filename, forests_path, sep2frags_path, batch_size, split):
        super(MyDataset).__init__()

        self.dset = DglGraphPropPredDataset(name=filename)
        split_idx = self.dset.get_idx_split()

        hfs = []
        s2fs = []
        for suf in range(3):
            hfs.append(h5py.File(forests_path.format(suf), 'r'))
            with open(sep2frags_path.format(suf), 'rb') as f:
                s2fs += pickle.load(f)
        grs = np.concatenate([hf['grs'][:] for hf in hfs])
        prs = np.concatenate([hf['prs'][:] for hf in hfs])
        _ = [hf.close() for hf in hfs]

        def data_el(i):
            d = {'gr': grs[i], 'pr': prs[i], 'sep2frag': s2fs[i], 'label': self.dset[i][1]}
            mol = self.dset[i][0]
            mol = Molecule(mol.nodes().numpy(), np.array(list(map(np.array,
                mol.edges()))).T, mol.ndata['feat'].numpy(), mol.edata['feat'].numpy())
            d['nfeat'], d['efeat'] = mol.nfeat, mol.efeat

            # Map (u, v) edge to index in d['efeat']
            d['e2i'] = {tuple(v): k for k, v in enumerate(mol.edge_list)}
            d['e_out'] = [t for t in product(range(mol.nfeat.shape[0]),
                repeat=2) if t[0] != t[1] and t not in d['e2i']]
            d['e_out'] = np.stack(d['e_out']) if d['e_out'] else None
            return d

        self.data = [data_el(i) for i in split_idx[split]]
        print(f'Get {len(self.data)} molecule-label pairs from {filename}')

        self.batch_size = int(batch_size)
        self.train = split == 'train'

    def __iter__(self):
        idx = list(range(len(self.data)))

        if self.train:
            random.shuffle(idx)
            idx.sort(key=lambda i: self.data[i]['nfeat'].shape[0])

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += self.data[i]['nfeat'].shape[0]
            data.append(self.data[i])
            if num_tokens >= self.batch_size or len(data) > 256:
                batches.append(data)
                num_tokens, data = 0, []

        if self.train:
            random.shuffle(batches)

        for batch in batches:
            yield batchify(batch, self.train)

def parse_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='../data/AMR/amr_2.0/dev.txt.features.preproc.json')
    parser.add_argument('--train_batch_size', type=int, default=10)

    return parser.parse_args()

if __name__ == '__main__':
    pass

