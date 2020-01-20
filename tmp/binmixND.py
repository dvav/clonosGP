import numpy as nmp
import theano.tensor as tns
import pymc3 as pmc

import GPyclone.aux as aux

##
# def get_model(r, R, vaf0, K=10, sigma=0.1):
#     nsamples = r.shape[1]
#     r, R, vaf0 = r[:, :, None], R[:, :, None], vaf0[:, :, None]
#     idxs = aux.corr_vector_to_matrix_indices(K)
#     S = tns.eye(K) * sigma**2
#     with pmc.Model() as model:
#         w = pmc.Dirichlet('w', nmp.ones(K))
#         lw = tns.log(w)

#         # alpha = pmc.Gamma('alpha', 1.0, 1.0)
#         # u = pmc.Beta('u', 1.0, alpha, shape=K-1)
#         # lw = aux.stick_breaking_log(u)

#         C = tns.fill_diagonal(pmc.LKJCorr('C', eta=2.0, n=K)[idxs], 1.0)
#         psi = pmc.MvNormal('psi', mu=nmp.zeros(K), cov=C+S, shape=(nsamples, K))
#         phi = pmc.Deterministic('phi', pmc.invlogit(psi))

#         # psi = pmc.MvNormal('psi', mu=nmp.zeros(K), tau=nmp.eye(K), shape=(nsamples, K))    
#         # phi = pmc.Deterministic('phi', pmc.invlogit(psi))

#         theta = pmc.Deterministic('theta', vaf0 * phi[None, :, :])
#         pmc.DensityDist('r', aux.binmixND_logp_fcn(R, theta, lw), observed=r)
#     return model


# def get_model(data, K, alpha, sigma, sigma2, eta, *args, **kargs):
#     r = data.pivot(index='MUTID', columns='SAMPLEID', values='r').values
#     R = data.pivot(index='MUTID', columns='SAMPLEID', values='R').values
#     VAF0 = data.pivot(index='MUTID', columns='SAMPLEID', values='VAF0').values
#     r, R, VAF0 = r[:, :, None], R[:, :, None], VAF0[:, :, None]

#     nsamples = data.SAMPLEID.nunique()
    
#     idxs = aux.corr_vector_to_matrix_indices(K)
#     D = tns.eye(K) * sigma**2   
#     with pmc.Model() as model:
#         # w = pmc.Dirichlet('w', nmp.ones(K))
#         # lw = tns.log(w)

#         # alpha = pmc.Gamma('alpha', 1.0, 1.0)
#         u = pmc.Beta('u', 1.0, alpha, shape=K-1)
#         lw = pmc.Deterministic('lw', aux.stick_breaking_log(u))

#         C_ = pmc.LKJCorr('C', eta=eta, n=K)
#         C = tns.fill_diagonal(C_[idxs], 1.0)
#         mu_psi = pmc.MvNormal('mu_psi', mu=nmp.zeros(K), cov=C, shape=(nsamples, K))
#         psi = pmc.Normal('psi', mu=mu_psi, sd=sigma2, shape=(nsamples, K))
#         phi = pmc.Deterministic('phi', pmc.invlogit(psi))

#         # psi = pmc.MvNormal('psi', mu=nmp.zeros(K), cov=D, shape=(nsamples, K))  
#         # phi = pmc.Deterministic('phi', pmc.invlogit(psi))

#         theta = pmc.Deterministic('theta', VAF0 * phi[None, :, :])
#         pmc.DensityDist('r', aux.binmixND_logp_fcn(R, theta, lw), observed=r)
#     return model


# def get_model(data, K, alpha, sigma, sigma2, eta, *args, **kargs):
#     r = data.pivot(index='MUTID', columns='SAMPLEID', values='r').values
#     R = data.pivot(index='MUTID', columns='SAMPLEID', values='R').values
#     VAF0 = data.pivot(index='MUTID', columns='SAMPLEID', values='VAF0').values
#     r, R, VAF0 = r[:, :, None], R[:, :, None], VAF0[:, :, None]

#     nsamples = data.SAMPLEID.nunique()
    
#     idxs = aux.corr_vector_to_matrix_indices(K)
#     D = tns.eye(K) * sigma**2
#     with pmc.Model() as model:
#         w = pmc.Dirichlet('w', nmp.ones(K))
#         lw = tns.log(w)

#         # alpha = pmc.Gamma('alpha', 1.0, 1.0)
#         # u = pmc.Beta('u', 1.0, alpha, shape=K-1)
#         # lw = pmc.Deterministic('lw', aux.stick_breaking_log(u))

#         C_ = pmc.LKJCorr('C', eta=eta, n=K)
#         C = tns.fill_diagonal(C_[idxs], 1.0)
#         psi = pmc.MvNormal('psi', mu=nmp.zeros(K), cov=C, shape=(nsamples, K))
#         phi = pmc.Deterministic('phi', pmc.invlogit(psi))

#         # psi = pmc.MvNormal('psi', mu=nmp.zeros(K), cov=D, shape=(nsamples, K))  
#         # phi = pmc.Deterministic('phi', pmc.invlogit(psi))

#         theta = pmc.Deterministic('theta', VAF0 * phi[None, :, :])
#         pmc.DensityDist('r', aux.binmixND_logp_fcn(R, theta, lw), observed=r)
#     return model


def get_model(data, K, alpha, sigma, sigma2, eta, *args, **kargs):
    r = data.pivot(index='MUTID', columns='SAMPLEID', values='r').values
    R = data.pivot(index='MUTID', columns='SAMPLEID', values='R').values
    VAF0 = data.pivot(index='MUTID', columns='SAMPLEID', values='VAF0').values
    r, R, VAF0 = r[:, :, None], R[:, :, None], VAF0[:, :, None]

    nsamples = data.SAMPLEID.nunique()
    
    idxs = aux.corr_vector_to_matrix_indices(nsamples)
    D = tns.eye(nsamples) * sigma**2
    with pmc.Model() as model:
        # alpha = pmc.Gamma('alpha', 1.0, 1.0)
        u = pmc.Beta('u', 1.0, alpha, shape=K-1)
        lw = pmc.Deterministic('lw', aux.stick_breaking_log(u))

        C_ = pmc.LKJCorr('C', eta=eta, n=nsamples)
        C = tns.fill_diagonal(C_[idxs], 1.0)
        Sigma = D.dot(C)
        psi = pmc.MvNormal('psi', mu=nmp.zeros(nsamples), cov=Sigma, shape=(K, nsamples))
        phi = pmc.Deterministic('phi', pmc.invlogit(psi.T))

        # psi = pmc.MvNormal('psi', mu=nmp.zeros(nsamples), cov=D, shape=(K, nsamples))
        # phi = pmc.Deterministic('phi', pmc.invlogit(psi.T))

        theta = pmc.Deterministic('theta', VAF0 * phi[None, :, :])

        pmc.DensityDist('r', aux.binmixND_logp_fcn(R, theta, lw), observed=r)
    return model
