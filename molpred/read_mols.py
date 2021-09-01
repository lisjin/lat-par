#!/usr/bin/env python3

import argparse
import h5py
import pandas as pd
import numpy as np

from ogb.graphproppred import DglGraphPropPredDataset
#from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from scipy import sparse as sp
from rdkit.Chem import MolFromSmiles

def read_smiles():
	smiles = pd.read_csv(f'dataset/ogbg_molhiv/mapping/mol.csv.gz',
			compression='gzip', header=None)[1][1:].to_numpy()
	atoms = [[a.GetAtomicNum() for a in MolFromSmiles(sm).GetAtoms()] for sm in smiles]
	return atoms

def main(args):
	dat = DglGraphPropPredDataset(name=args.dset_name)
	import pdb; pdb.set_trace()
	f = h5py.File(f'{args.dset_name}.hdf5', 'w')
	dt = h5py.vlen_dtype(np.dtype('int32'))
	grs = f.create_dataset('grs', (len(dat),), dtype=dt)
	for i, d in enumerate(dat):
		grs[i] = np.array(sp.find(d[0].adj(scipy_fmt='coo'))[:2]).reshape(-1)
	f.close()

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('--dset_name', type=str, default='ogbg-molhiv')
	args = ap.parse_args()
	main(args)
