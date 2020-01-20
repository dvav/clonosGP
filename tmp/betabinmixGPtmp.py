import numpy as nmp
import pandas as pnd
import theano.tensor as tns
import pymc3 as pmc

from GPyclone import aux

# ## Full covariance
# def get_model(data, K, alpha, eta, *args, **kargs):
#     r = data.pivot(index='MUTID', columns='SAMPLEID', values='r').values
#     R = data.pivot(index='MUTID', columns='SAMPLEID', values='R').values
#     VAF0 = data.pivot(index='MUTID', columns='SAMPLEID', values='VAF0').values
#     r, R, VAF0 = r[:, :, None], R[:, :, None], VAF0[:, :, None]

#     nsamples = data.SAMPLEID.nunique()
#     idxs = aux.corr_vector_to_matrix_indices(nsamples)
#     with pmc.Model() as mdl:
#         u = pmc.Beta('u', 1.0, alpha, shape=K-1, testval=0.5)
#         lw = pmc.Deterministic('lw', aux.stick_breaking_log(u))
#         L_ = pmc.LKJCholeskyCov('L', eta=eta, n=nsamples, sd_dist=pmc.InverseGamma.dist(1.0, 1.0))
#         L = pmc.expand_packed_triangular(nsamples, L_, lower=True)
#         # C_ = pmc.LKJCorr('C', eta=eta, n=nsamples, testval=nmp.zeros(int(nsamples*(nsamples-1)/2)))
#         # C = tns.fill_diagonal(C_[idxs], scale**2)
#         # L = tns.slinalg.cholesky(C)
#         psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(nsamples, K), testval=0.0)
#         phi = pmc.Deterministic('phi', pmc.invlogit(L.dot(psi)))
#         theta = pmc.Deterministic('theta', VAF0 * phi[None, :, :])
#         s = pmc.Uniform('s', shape=nsamples, testval=0.5)
#         pmc.DensityDist('r', aux.betabinmixND_logp_fcn(R, ((1.0 - s) / s)[:, None], theta, lw), observed=r)
    
#     #
#     return mdl


## Squared exponential 1
# def get_model(data, K, alpha, cov, *args, **kargs):
#     t = data[['SAMPLEID', 'TIME']].drop_duplicates().TIME.values
#     r, R, VAF0 = data.r.values, data.R.values, data.VAF0.values
#     nmuts, nsamples = data.MUTID.nunique(), data.SAMPLEID.nunique()
    
#     r = r.reshape(nsamples, nmuts).T[:, :, None]
#     R = R.reshape(nsamples, nmuts).T[:, :, None]
#     VAF0 = VAF0.reshape(nsamples, nmuts).T[:, :, None]

#     t = (t - nmp.min(t)) / (nmp.max(t) - nmp.min(t))      # map time to the [0,1] interval
#     with pmc.Model() as mdl:
#         v = pmc.Beta('v', 1.0, alpha, shape=K-1, testval=0.5)
#         lw = pmc.Deterministic('lw', aux.stick_breaking_log(v))        
#         tau = pmc.Gamma('tau', 1.0, 1.0, testval=1.0)
#         h2 = pmc.Gamma('h2', 1.0, 1.0, testval=1.0)
#         if cov == 'ExpQuad':
#             S = aux.cov_expquad(t, h2, tau)
#         elif cov == 'Exp':
#             S = aux.cov_exp(t, h2, tau)
#         elif cov == 'Mat32':
#             S = aux.cov_mat32(t, h2, tau)
#         elif cov == 'Mat52':
#             S = aux.cov_mat52(t, h2, tau)                        
#         else:
#             raise Exception(f'Unknown covariance function: {cov}')
#         L = tns.slinalg.cholesky(S)
#         psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(nsamples, K), testval=0.0)
#         phi = pmc.Deterministic('phi', pmc.invlogit(L.dot(psi)))
#         s = pmc.Uniform('s', shape=nsamples, testval=0.5)
#         pmc.DensityDist('r', aux.betabinmixND_logp_fcn(R, ((1.0 - s) / s)[:, None], VAF0 * phi[None, :, :], lw), observed=r)

#     #
#     return mdl


# ## Squared exponential 2
# def get_model(data, K, alpha, cov, *args, **kargs):
#     t = data[['SAMPLEID', 'TIME2']].drop_duplicates().TIME2.values
#     r, R, VAF0 = data.r.values, data.R.values, data.VAF0.values
#     nmuts, nsamples = data.MUTID.nunique(), data.SAMPLEID.nunique()
    
#     r = r.reshape(nsamples, nmuts).T[:, :, None]
#     R = R.reshape(nsamples, nmuts).T[:, :, None]
#     VAF0 = VAF0.reshape(nsamples, nmuts).T[:, :, None]

#     with pmc.Model() as mdl:
#         v = pmc.Beta('v', 1.0, alpha, shape=K-1, testval=0.5)
#         lw = pmc.Deterministic('lw', aux.stick_breaking_log(v))        
#         tau = pmc.Gamma('tau', 1.0, 1.0, testval=1.0)
#         h2 = pmc.Gamma('h2', 1.0, 1.0, shape=K, testval=1.0)
#         # S = aux.cov_expquad(t, t, 1.0, tau) + tns.eye(nsamples) * 1e-6
#         # S = aux.cov_exp(t, t, 1.0, tau) + tns.eye(nsamples) * 1e-6
#         S = aux.cov_mat32(t, t, 1.0, tau) + tns.eye(nsamples) * 1e-6
#         # S = aux.cov_mat52(t, t, 1.0, tau) + tns.eye(nsamples) * 1e-6                        
#         Lr = tns.slinalg.cholesky(S)
#         Lc = tns.diag(tns.sqrt(h2))
#         psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(nsamples, K), testval=0.0)
#         phi = pmc.Deterministic('phi', pmc.invlogit(Lr.dot(psi).dot(Lc.T)))
#         s = pmc.Uniform('s', shape=nsamples, testval=0.5)
#         pmc.DensityDist('r', aux.betabinmixND_logp_fcn(R, ((1.0 - s) / s)[:, None], VAF0 * phi[None, :, :], lw), observed=r)

#     #
#     return mdl

## Squared exponential 3
def get_model(data, K, alpha, *args, **kargs):
    t = data[['SAMPLEID', 'TIME2']].drop_duplicates().TIME2.values
    r, R, VAF0 = data.r.values, data.R.values, data.VAF0.values
    nmuts, nsamples = data.MUTID.nunique(), data.SAMPLEID.nunique()
    
    r = r.reshape(nsamples, nmuts).T[:, :, None]
    R = R.reshape(nsamples, nmuts).T[:, :, None]
    VAF0 = VAF0.reshape(nsamples, nmuts).T[:, :, None]

    M = tns.slinalg.kron(tns.eye(K), aux.cov_mat32(t, t, 1.0, 1.0))
    with pmc.Model() as mdl:
        v = pmc.Beta('v', 1.0, alpha, shape=K-1, testval=0.5)
        lw = pmc.Deterministic('lw', aux.stick_breaking_log(v))        
        tau = pmc.Gamma('tau', 1.0, 1.0, shape=K, testval=1.0)
        h2 = pmc.Gamma('h2', 1.0, 1.0, shape=K, testval=1.0)
        Q = tns.repeat(h2, nsamples)*M**tns.repeat(tau, nsamples) + tns.eye(nsamples*K)*1e-6
        L = tns.slinalg.cholesky(Q)
        psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=nsamples*K, testval=0.0)
        phi = pmc.Deterministic('phi', pmc.invlogit(L.dot(psi)).reshape((K, nsamples)).T)
        s = pmc.Uniform('s', shape=nsamples, testval=0.5)
        pmc.DensityDist('r', aux.betabinmixND_logp_fcn(R, ((1.0 - s) / s)[:, None], VAF0 * phi[None, :, :], lw), observed=r)

    #
    return mdl

## Squared exponential 4
# def get_model(data, K, alpha, eta, *args, **kargs):
#     t = data[['SAMPLEID', 'TIME2']].drop_duplicates().TIME2.values
#     r, R, VAF0 = data.r.values, data.R.values, data.VAF0.values
#     nmuts, nsamples = data.MUTID.nunique(), data.SAMPLEID.nunique()
    
#     r = r.reshape(nsamples, nmuts).T[:, :, None]
#     R = R.reshape(nsamples, nmuts).T[:, :, None]
#     VAF0 = VAF0.reshape(nsamples, nmuts).T[:, :, None]

#     M = nmp.exp(-0.5*(t-t[:, None])**2)
#     with pmc.Model() as mdl:
#         v = pmc.Beta('v', 1.0, alpha, shape=K-1, testval=0.5)
#         lw = pmc.Deterministic('lw', aux.stick_breaking_log(v)) 
#         tau = pmc.Gamma('tau', 1.0, 1.0, testval=1.0)
#         Lr = tns.slinalg.cholesky(M**tau + tns.eye(nsamples)*1e-6)
#         # a = pmc.Normal('a', shape=(K, 1), testval=0.0)
#         # h2 = pmc.Gamma('h2', 1.0, 1.0, shape=K, testval=1.0)
#         # Lc = tns.slinalg.cholesky(tns.dot(a, a.T) + tns.diag(1.0/tns.sqrt(h2)))
#         Lc_ = pmc.LKJCholeskyCov('L', eta=eta, n=K, sd_dist=pmc.Gamma.dist(1.0, 1.0, shape=K))
#         Lc = pmc.expand_packed_triangular(K, Lc_, lower=True)
#         psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(nsamples, K), testval=0.0)
#         phi = pmc.Deterministic('phi', pmc.invlogit(Lr.dot(psi).dot(Lc.T)))
#         s = pmc.Uniform('s', shape=nsamples, testval=0.5)
#         pmc.DensityDist('r', aux.betabinmixND_logp_fcn(R, ((1.0 - s) / s)[:, None], VAF0 * phi[None, :, :], lw), observed=r)

#     #
#     return mdl

## Linear kernel
# def get_model(data, K, alpha, *args, **kargs):
#     t = data[['SAMPLEID', 'TIME']].drop_duplicates().TIME.values
#     r, R, VAF0 = data.r.values, data.R.values, data.VAF0.values
#     nmuts, nsamples = data.MUTID.nunique(), data.SAMPLEID.nunique()
    
#     r = r.reshape(nsamples, nmuts).T[:, :, None]
#     R = R.reshape(nsamples, nmuts).T[:, :, None]
#     VAF0 = VAF0.reshape(nsamples, nmuts).T[:, :, None]

#     t = (t - nmp.min(t)) / (nmp.max(t) - nmp.min(t))       # map time to the [0,1] interval
#     dt = nmp.r_[1.0, nmp.diff(t)]
#     Lr = nmp.tril([dt]*nsamples)
#     with pmc.Model() as mdl:
#         v = pmc.Beta('v', 1.0, alpha, shape=K-1, testval=0.5)
#         lw = pmc.Deterministic('lw', aux.stick_breaking_log(v))
#         tau = pmc.Gamma('tau', 1.0, 1.0, shape=K, testval=1.0)
#         psi = pmc.Normal('psi', mu=0.0, sd=1.0, shape=(nsamples, K), testval=0.0)
#         Lc = tns.diag(1.0/tns.sqrt(tau))
#         phi = pmc.Deterministic('phi', pmc.invlogit(tns.dot(Lr, psi).dot(Lc.T)))
#         s = pmc.Uniform('s', shape=nsamples, testval=0.5)
#         pmc.DensityDist('r', aux.betabinmixND_logp_fcn(R, ((1.0 - s) / s)[:, None], VAF0 * phi[None, :, :], lw), observed=r)

#     #
#     return mdl


# def calculate_clusters(data, trace, loc=nmp.mean, alpha=0.05):
#     w_samples = nmp.exp(trace['lw'])
#     phi_samples = trace['phi']
    
#     w = loc(w_samples, 0)
#     w_lo, w_hi = nmp.percentile(w_samples, [0.5*alpha*100, (1 - 0.5*alpha)*100], axis=0) 

#     t = nmp.linspace(data.TIME2.min(), data.TIME2.max(), num=100)
#     tau = nmp.mean(trace.tau, 0)
#     h2 = nmp.mean(trace.h2, 0)
#     M = nmp.exp(-0.5 * (t - t[:, None])**2)
#     phi = [ for tau_, h2_ in zip(tau, h2)]
#     phi_lo, phi_hi = 0*nmp.percentile(phi_samples, [0.5*alpha*100, (1 - 0.5*alpha)*100], axis=0)
    
#     clusters = pnd.DataFrame({'CLUSTERID': nmp.arange(w.size)+1, 'W': w, 'W_LO': w_lo, 'W_HI':w_hi})
#     clusters['PHI'], clusters['PHI_LO'], clusters['PHI_HI'] = list(phi.T), list(phi_lo.T), list(phi_hi.T)

#     #
#     return clusters
