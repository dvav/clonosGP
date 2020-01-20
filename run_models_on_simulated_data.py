import os
import sys

IDX = int(sys.argv[1])
os.environ['THEANO_FLAGS'] = f'base_compiledir="theano/p{IDX}"'

import tqdm as tqd
import itertools as itr
import numpy as nmp
import pandas as pnd
import sklearn.metrics as mtr
import scipy.special as scp

import pymc3 as pmc
import clonosGP as cln



##
def run_model(prior, cov, lik, R, K, M, N, tau, h2, data):    
    nmp.random.seed(42)
    pmc.tt_rng(42)
    
    res = cln.infer(data, 
                    model_args={'K': 20, 'prior': prior, 'cov': cov, 'lik': lik, 'threshold': 0.0}, 
                    pymc3_args={'niters': 40000, 'method': 'advi', 'flow': 'scale-loc', 'learning_rate': 1e-2, 'random_seed': 42})

    z_true = data[['MUTID', 'CLUSTERID']].drop_duplicates().CLUSTERID.values
    z_pred = res['data'][['MUTID', 'CLUSTERID']].drop_duplicates().CLUSTERID.values
        
    return pnd.DataFrame({
        'REP': R, 
        'NCLUSTERS': K, 
        'NSAMPLES': M, 
        'NMUTS': N,
        'TAU': tau,
        'H2': h2,
        'PRIOR': prior,
        'COV': cov,
        'LIK': lik,
        'ARI': mtr.adjusted_rand_score(z_true, z_pred),
        'AMI': mtr.adjusted_mutual_info_score(z_true, z_pred),
        'FMI': mtr.fowlkes_mallows_score(z_true, z_pred),
    }, index=[0]).reset_index(drop=True)


## load template
depths = pnd.read_csv('data/cll_Rincon_2019_patient1.csv').R.values

##
print(f'Generating data: {IDX}')
nmp.random.seed(42)
DATA = [[R, K, M, N, TAU, H2, cln.sim.get_Nsamples(nclusters=K, nmuts=N, nsamples=M, tau=TAU, h2=H2, mean_depth=(40, 40), depths=depths)] 
        for R, K, M, N, TAU, H2 in itr.product([1, 2, 3], [2, 4, 8], [3, 6, 12], [25, 50, 100], [1, 10, 100], [1, 10, 20])]

##
print(f"Running model: {IDX}")
ARGS = [('Flat', 'Exp', 'Bin'), ('Flat', 'Exp', 'BBin'), ('GP0', 'Exp', 'Bin'), ('GP0', 'Exp', 'BBin'),('GP0', 'ExpQ', 'Bin'),('GP0', 'ExpQ', 'BBin')]
RES = [run_model(*args, *DATA[IDX-1]) for args in ARGS]
RES = pnd.concat(RES).reset_index(drop=True)
RES.to_csv(f'results/simdata{IDX}.csv', index=False)


