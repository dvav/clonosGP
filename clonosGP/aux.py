import logging
logger = logging.getLogger('clonosGP')

import numpy as nmp
import scipy as scp
import pandas as pnd
import theano.tensor as tns
import pymc3 as pmc


##
def calculate_multiplicity(vcf, rho, CNt, CNn):
    u = vcf * (rho * CNt + (1.0 - rho) * CNn) / rho
    # m = nmp.where(u < 1, 1, nmp.rint(u).astype('int'))
    m = nmp.clip(nmp.rint(u).astype('int'), 1, CNt)

    return m


##
def calculate_vaf0(rho, CNm, CNt, CNn):
    return rho * CNm / (rho * CNt + (1.0 - rho) * CNn)


##
def corr_vector_to_matrix_indices(D):
    n = D * (D - 1) / 2
    idxs = nmp.zeros((D, D), dtype='int')
    idxs[nmp.triu_indices(D, k=1)] = nmp.arange(n)
    idxs[nmp.triu_indices(D, k=1)[::-1]] = nmp.arange(n) 
    
    return idxs


##
def prepare_data(data):
    data = data.copy(deep=True)
    
    #
    if not isinstance(data, pnd.DataFrame): 
        raise Exception('Input data should be a DataFrame in long format.')
    
    #
    cols = ['SAMPLEID', 'MUTID', 'r', 'R']
    if not nmp.all([el in data.columns for el in cols]):
        raise Exception(f'Input data should include at least the following columns: {cols}')

    #
    if 'PURITY' not in data.columns:
        logger.info('No PURITY column in the data. Assuming all samples have purity 100%.')
        data['PURITY'] = 1.0
        
    #
    if data.SAMPLEID.nunique() > 1:   # data has multiple samples
        if 'TIME' in data.columns:
            data['TIME2'] = (data.TIME - data.TIME.min()) / (data.TIME.max() - data.TIME.min())      # map time to the [0,1] interval
        else:
            logger.info('Multiple samples detected in the data, but there is no TIME column. Assuming data is cross-senctional.')

    #
    if 'CNn' not in data.columns:
        logger.info('No CNn column in the data. Assuming germline is diploid over all provided loci.')
        data['CNn'] = 2

    #
    if 'CNt' not in data.columns:
        logger.info('No CNt column in the data. Assuming all tumour samples are diploid over all provided loci.')
        data['CNt'] = 2

    # 
    if 'CNm' not in data.columns:
        logger.info('No CNm column in the data. Multiplicity values will be approximated.')
        data['CNm'] = calculate_multiplicity(data.r/data.R, data.PURITY, data.CNt, data.CNn)

    #
    data['VAF'] = data.r / data.R
    data['VAF0'] = calculate_vaf0(data.PURITY, data.CNm, data.CNt, data.CNn)
    data['CCF_'] = nmp.clip(data.VAF/data.VAF0, 0.0, 1.0)

    #
    if nmp.any(data.PURITY <= 0) or nmp.any(data.PURITY > 1):
        raise Exception('Tumour purity values should be above 0 and less than or equal to 1.')      

    #
    if nmp.any(data.r > data.R):
        raise Exception('Variant allele reads r should not be larger than the total number of reads R.')        

    #
    if nmp.any(data.R <= 0) or nmp.any(data.CNn <= 0) or nmp.any(data.CNt <= 0):
        raise Exception('Coverage, normal and tumour local copy numbers should be strictly positive.')        

    #
    if nmp.any(data.CNm <= 0) or nmp.any(data.CNm > data.CNt):
        raise Exception('Multiplicity values cannot be negative or larger than the tumour local copy number.')        

    #
    if 'TIME2' in data.columns:
        samples = data[['SAMPLEID', 'PURITY', 'TIME', 'TIME2']].drop_duplicates().sort_values('TIME2').reset_index(drop=True)
    else: 
        samples = data[['SAMPLEID', 'PURITY']].drop_duplicates().reset_index(drop=True)

    #
    if nmp.any(data[['SAMPLEID', 'MUTID', 'r', 'R', 'CNn', 'CNt', 'CNm', 'VAF', 'VAF0', 'CCF_']].isna()):
        raise Exception('Missing values are not allowed in the data.')    

    #
    wdata = data[['SAMPLEID', 'MUTID', 'r', 'R', 'CNn', 'CNt', 'CNm', 'VAF', 'VAF0', 'CCF_']]
    wdata = wdata.pivot(index='MUTID', columns='SAMPLEID', values=['r', 'R', 'CNn', 'CNt', 'CNm', 'VAF', 'VAF0', 'CCF_'])

    #
    if nmp.any(wdata.isna()):
        raise Exception('Missing values are not allowed in the data. Check whether all loci are covered in all samples.')    

    #
    return {
        'samples': samples,
        'r': wdata.r[samples.SAMPLEID],
        'R': wdata.R[samples.SAMPLEID],
        'CNn': wdata.CNn[samples.SAMPLEID],
        'CNt': wdata.CNt[samples.SAMPLEID],
        'CNm': wdata.CNm[samples.SAMPLEID],
        'VAF': wdata.VAF[samples.SAMPLEID],
        'VAF0': wdata.VAF0[samples.SAMPLEID],
        'CCF_': wdata.CCF_[samples.SAMPLEID]
    }


##
def pivot_longer(data):
    df = pnd.concat({_: data[_] for _ in ['r', 'R', 'CNn', 'CNt', 'CNm', 'VAF', 'VAF0', 'CCF_']}, axis=1).stack().reset_index()
    df = pnd.merge(data['samples'], df)

    #
    return df