import numpy as nmp
import pandas as pnd
import scipy.special as scp
import theano.tensor as tns
import pymc3 as pmc

import clonosGP.stats as sts
import clonosGP.aux as aux

##
def get_model(data, prior, lik, cov, *args, **kargs):
    r, R, VAF0 = data['r'].values.T, data['R'].values.T, data['VAF0'].values.T
    nsamples, _ = r.shape

    cov_fcn = sts.COV_FCNS[cov]

    if (nsamples == 1) or (prior == 'Flat') or ('TIME2' not in data['samples']):
        if lik == 'BBin':
            mdl = get_model_Flat_BetaBin(r, R, VAF0, *args, **kargs)
        elif lik == 'Bin':
            mdl = get_model_Flat_Bin(r, R, VAF0, *args, **kargs)
        else:
            raise Exception(f'Unknown likelihood function: {lik}')
    else:   # assuming prior is 'GP*' and 'TIME2' in data
        t = data['samples'].TIME2.values.ravel()
        if (prior == 'GP0') and (lik == 'BBin'):
            mdl = get_model_GP0_BetaBin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)        
        elif (prior == 'GP1') and (lik == 'BBin'):
            mdl = get_model_GP1_BetaBin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)
        elif (prior == 'GP2') and (lik == 'BBin'):
            mdl = get_model_GP2_BetaBin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)
        elif (prior == 'GP3') and (lik == 'BBin'):
            mdl = get_model_GP3_BetaBin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)
        elif (prior == 'GP0') and (lik == 'Bin'):
            mdl = get_model_GP0_Bin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)
        elif (prior == 'GP1') and (lik == 'Bin'):
            mdl = get_model_GP1_Bin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)
        elif (prior == 'GP2') and (lik == 'Bin'):
            mdl = get_model_GP2_Bin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)
        elif (prior == 'GP3') and (lik == 'Bin'):
            mdl = get_model_GP3_Bin(t, r, R, VAF0, cov_fcn=cov_fcn, *args, **kargs)
        else:
            raise Exception(f'Unknown prior ({prior}) or likelihood function ({lik}) requested.')
    #
    return mdl


##
def get_model_DP(K, alpha):
    if alpha is None:
        alpha_ = pmc.Uniform('alpha_', testval=0.5)
        alpha = pmc.Deterministic('alpha', (1.0 - alpha_)/alpha_)
    _ = pmc.Beta('v', 1.0, alpha, shape=K-1, testval=0.5)
    lw = pmc.Deterministic('lw', sts.stick_breaking_log(_))
    return lw


##
def get_model_Flat(K, nsamples):
    phi = pmc.Uniform('phi', shape=(K, nsamples), testval=0.5)
    return phi


##
def get_model_GP0(t, K, nsamples, cov_fcn):
    tau = pmc.Gamma('tau', 1.0, 1.0, testval=1.0)
    h2 = pmc.Gamma('h2', 1.0, 1.0, testval=1.0)
    S =  h2 * cov_fcn(t, t[:, None], tau) + tns.eye(nsamples) * 1e-6
    L = tns.slinalg.cholesky(S)
    psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(K, nsamples), testval=0.0)
    phi = pmc.Deterministic('phi', pmc.invlogit(tns.dot(psi, L.T)))

    return phi


##
def get_model_GP1(t, K, nsamples, cov_fcn):
    tau = pmc.Gamma('tau', 1.0, 1.0, testval=1.0)
    h2 = pmc.Gamma('h2', 1.0, 1.0, shape=K, testval=1.0)
    S =  cov_fcn(t, t[:, None], tau) + tns.eye(nsamples)*1e-6
    Lc = tns.slinalg.cholesky(S)
    Lr = tns.diag(tns.sqrt(h2))
    psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(K, nsamples), testval=0.0)
    phi = pmc.Deterministic('phi', pmc.invlogit(Lr.dot(psi).dot(Lc.T)))
    return phi


##
def get_model_GP2(t, K, nsamples, cov_fcn, eta):
    tau = pmc.Gamma('tau', 1.0, 1.0, testval=1.0)
    S =  cov_fcn(t, t[:, None], tau) + tns.eye(nsamples)*1e-6
    Lc = tns.slinalg.cholesky(S)
    Lr_ = pmc.LKJCholeskyCov('Lr_', eta=eta, n=K, sd_dist=pmc.Gamma.dist(1.0, 1.0, shape=K))
    Lr = pmc.expand_packed_triangular(K, Lr_, lower=True)
    psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(K, nsamples), testval=0.0)
    phi = pmc.Deterministic('phi', pmc.invlogit(Lr.dot(psi).dot(Lc.T)))

    pmc.Deterministic('h2', tns.diag(Lr.dot(Lr.T)))    

    return phi


##
def get_model_GP3(t, K, nsamples, cov_fcn):
    M = tns.slinalg.kron(tns.eye(K), cov_fcn(t, t[:, None], 1.0))    
    tau = pmc.Gamma('tau', 1.0, 1.0, shape=K, testval=1.0)
    h2 = pmc.Gamma('h2', 1.0, 1.0, shape=K, testval=1.0)
    Q = tns.repeat(h2, nsamples)*M**tns.repeat(tau, nsamples) + tns.eye(nsamples*K)*1e-6
    L = tns.slinalg.cholesky(Q)
    psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=K*nsamples, testval=0.0)
    phi = pmc.Deterministic('phi', pmc.invlogit(L.dot(psi)).reshape((K, nsamples)))
    return phi


##
def get_model_Flat_BetaBin(r, R, VAF0, K, alpha, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_Flat(K, nsamples) 
        s = pmc.Uniform('s', shape=nsamples, testval=0.5)
        u = pmc.Deterministic('u', (1.0 - s) / s)[None, :, None]  
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])
        pmc.DensityDist('r', sts.betabinmix_logp_fcn(R, u, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_Flat_Bin(r, R, VAF0, K, alpha, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_Flat(K, nsamples) 
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])
        pmc.DensityDist('r', sts.binmix_logp_fcn(R, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP0_BetaBin(t, r, R, VAF0, K, alpha, cov_fcn, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP0(t, K, nsamples, cov_fcn)
        s = pmc.Uniform('s', shape=nsamples, testval=0.5)
        u = pmc.Deterministic('u', (1.0 - s) / s)[None, :, None]
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])               
        pmc.DensityDist('r', sts.betabinmix_logp_fcn(R, u, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP0_Bin(t, r, R, VAF0, K, alpha, cov_fcn, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP0(t, K, nsamples, cov_fcn)
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])        
        pmc.DensityDist('r', sts.binmix_logp_fcn(R, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP1_BetaBin(t, r, R, VAF0, K, alpha, cov_fcn, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP1(t, K, nsamples, cov_fcn)
        s = pmc.Uniform('s', shape=nsamples, testval=0.5)
        u = pmc.Deterministic('u', (1.0 - s) / s)[None, :, None]
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])               
        pmc.DensityDist('r', sts.betabinmix_logp_fcn(R, u, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP1_Bin(t, r, R, VAF0, K, alpha, cov_fcn, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP1(t, K, nsamples, cov_fcn)
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])        
        pmc.DensityDist('r', sts.binmix_logp_fcn(R, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP2_BetaBin(t, r, R, VAF0, K, alpha, cov_fcn, eta, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP2(t, K, nsamples, cov_fcn, eta)
        s = pmc.Uniform('s', shape=nsamples, testval=0.5)
        u = pmc.Deterministic('u', (1.0 - s) / s)[None, :, None]  
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])              
        pmc.DensityDist('r', sts.betabinmix_logp_fcn(R, u, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP2_Bin(t, r, R, VAF0, K, alpha, cov_fcn, eta, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP2(t, K, nsamples, cov_fcn, eta)
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])        
        pmc.DensityDist('r', sts.binmix_logp_fcn(R, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP3_BetaBin(t, r, R, VAF0, K, alpha, cov_fcn, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP3(t, K, nsamples, cov_fcn)
        s = pmc.Uniform('s', shape=nsamples, testval=0.5)
        u = pmc.Deterministic('u', (1.0 - s) / s)[None, :, None]
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])        
        pmc.DensityDist('r', sts.betabinmix_logp_fcn(R, u, theta, lw[:, None]), observed=r)
    return mdl


##
def get_model_GP3_Bin(t, r, R, VAF0, K, alpha, cov_fcn, *args, **kargs):
    nsamples, _ = r.shape
    with pmc.Model() as mdl:
        lw = get_model_DP(K, alpha)
        phi = get_model_GP3(t, K, nsamples, cov_fcn)
        theta = pmc.Deterministic('theta', VAF0 * phi[:, :, None])        
        pmc.DensityDist('r', sts.binmix_logp_fcn(R, theta, lw[:, None]), observed=r)
    return mdl
